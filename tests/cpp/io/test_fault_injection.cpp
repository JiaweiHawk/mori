// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Fault catalog for the ibverbs data path (MORI-IO fault simulation, part 1/N).
//
// Each entry is one real-world failure written as a FaultRule, plus the transfer
// it hits. Every entry is held to the same four checks:
//   ends      the faulted transfer reaches SUCCESS or an error within kDeadlineMs
//   honest    it reports SUCCESS only if the data landed and, with notifications
//             on, the target was told
//   recovers  once the fault clears, a new transfer between the same engines succeeds
//   alive     the process neither exits nor crashes
// The checks describe the behavior we want, not today's: a failing entry is an
// open bug for a follow-up fix. Each entry runs in its own child process, so a
// crash or a hang in one does not stop the rest.
//
// Usage: test_fault_injection [id-substring]
// Needs -DENABLE_IO_FAULT_INJECTION=ON, one active RDMA NIC and one GPU.

#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/io/io.hpp"
#include "src/io/rdma/backend_impl.hpp"
#include "src/io/rdma/fault_injector.hpp"

using namespace mori::io;

namespace {

/* ---------------------------------------------------------------------------------------------- */
/*                                          Fault catalog                                         */
/* ---------------------------------------------------------------------------------------------- */

enum class Op {
  Write,  // one 1 MiB RDMA write
  Read,   // one 1 MiB RDMA read
  Batch,  // BatchWrite of 2 non-adjacent segments, one per QP (qpPerTransfer=2)
};

struct Entry {
  const char* id;         // <fault kind>.<what it hits>[.<when>]
  const char* realWorld;  // what this stands for in production
  const char* rule;       // FaultRule spec (see fault_injector.hpp); "" = no fault
  Op op;
  bool notify;  // MORI notifications on: a SEND/RECV after the data tells the target
  int warmup;   // clean transfers before the fault is armed (0 = arm before connecting)
};

const Entry kCatalog[] = {
    {"none.write", "no fault (baseline)", "", Op::Write, true, 0},

    // Post fails: ibv_post_send / ibv_post_recv returns an error.
    {"post_send_fail.write", "SQ full or driver rejects the post", "post_send_fail:value=12",
     Op::Write, false, 0},
    {"post_send_fail.batch.second_qp", "post fails on QP 2 after QP 1 already posted",
     "post_send_fail:skip=1", Op::Batch, false, 0},
    {"post_send_fail.notify", "data posted, then the notification post fails",
     "post_send_fail:skip=1", Op::Write, true, 0},
    {"post_recv_fail.connect", "pre-posting receives fails while connecting", "post_recv_fail",
     Op::Write, true, 0},
    {"post_recv_fail.repost", "re-posting a used receive fails mid-run", "post_recv_fail",
     Op::Write, true, 1},

    // Error completion: the NIC reports the WR failed and moves the QP to ERR.
    {"cqe_error.write.retry_exc", "link down or peer gone during a write",
     "cqe_error:op=write:value=retry_exc", Op::Write, false, 0},
    {"cqe_error.write.retry_exc.mid_run", "same, after 10 good transfers",
     "cqe_error:op=write:value=retry_exc", Op::Write, false, 10},
    {"cqe_error.read.retry_exc", "link down or peer gone during a read",
     "cqe_error:op=read:value=retry_exc", Op::Read, false, 0},
    {"cqe_error.write.rem_access", "stale rkey: peer deregistered the memory",
     "cqe_error:op=write:value=rem_access", Op::Write, false, 0},
    {"cqe_error.read.rem_op", "peer rejected the operation", "cqe_error:op=read:value=rem_op",
     Op::Read, false, 0},
    {"cqe_error.write.loc_prot", "local buffer deregistered under an in-flight write",
     "cqe_error:op=write:value=loc_prot", Op::Write, false, 0},
    {"cqe_error.write.fatal", "NIC internal error on one WR", "cqe_error:op=write:value=fatal",
     Op::Write, false, 0},
    {"cqe_error.batch.one_qp", "one of two QPs fails mid-batch", "cqe_error:op=write", Op::Batch,
     false, 0},
    // Initiator side only: the injected error lands after the real SEND was
    // delivered, so the target is still told. cqe_drop.notify.recv covers the target side.
    {"cqe_error.notify.rnr", "target stopped re-posting receives (RNR retries exhausted)",
     "cqe_error:op=send:value=rnr_retry_exc", Op::Write, true, 0},

    // Lost completion: the WR finishes but its CQE never arrives.
    {"cqe_drop.write", "write completion lost", "cqe_drop:op=write", Op::Write, false, 0},
    {"cqe_drop.read", "read completion lost", "cqe_drop:op=read", Op::Read, false, 0},
    {"cqe_drop.batch.one_qp", "one QP's completion lost in a batch", "cqe_drop:op=write", Op::Batch,
     false, 0},
    {"cqe_drop.notify.recv", "target loses the notification completion", "cqe_drop:op=recv",
     Op::Write, true, 0},

    // QP error: the QP dies (QP_FATAL, NIC reset) and every WR on it flushes.
    {"qp_error.write", "QP dies before the first write", "qp_error", Op::Write, false, 0},
    {"qp_error.write.mid_run", "QP dies after 10 good transfers", "qp_error", Op::Write, false, 10},
    {"qp_error.read", "QP dies before a read", "qp_error", Op::Read, false, 0},
    {"qp_error.batch.one_qp", "one of two QPs dies", "qp_error", Op::Batch, false, 0},
};

/* ---------------------------------------------------------------------------------------------- */
/*                                         Loopback rig                                           */
/* ---------------------------------------------------------------------------------------------- */

constexpr size_t kBufBytes = 1 << 20;
constexpr size_t kSegBytes = 256 << 10;  // Batch: [0, 256K) and [512K, 768K)
constexpr size_t kSeg2Offset = 512 << 10;
constexpr int kDeadlineMs = 5000;
constexpr int kChildTimeoutMs = 60000;
constexpr const char* kInitiatorKey = "fault_initiator";

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;

struct TestSkip : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

const char* CodeName(StatusCode code) {
  switch (code) {
    case StatusCode::SUCCESS:
      return "SUCCESS";
    case StatusCode::INIT:
      return "INIT";
    case StatusCode::IN_PROGRESS:
      return "IN_PROGRESS";
    case StatusCode::ERR_INVALID_ARGS:
      return "ERR_INVALID_ARGS";
    case StatusCode::ERR_NOT_FOUND:
      return "ERR_NOT_FOUND";
    case StatusCode::ERR_RDMA_OP:
      return "ERR_RDMA_OP";
    case StatusCode::ERR_BAD_STATE:
      return "ERR_BAD_STATE";
    case StatusCode::ERR_GPU_OP:
      return "ERR_GPU_OP";
    default:
      return "?";
  }
}

int GetFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  socklen_t len = sizeof(addr);
  int port = -1;
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
      getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
    port = ntohs(addr.sin_port);
  }
  close(fd);
  return port;
}

struct GpuBuffer {
  MemoryDesc desc{};
  void* ptr{nullptr};

  explicit GpuBuffer(IOEngine* engine) {
    HIP_RUNTIME_CHECK(hipSetDevice(0));
    HIP_RUNTIME_CHECK(hipMalloc(&ptr, kBufBytes));
    desc = engine->RegisterMemory(ptr, kBufBytes, 0, MemoryLocationType::GPU);
  }
};

struct Transfer {
  TransferStatus status;
  TransferUniqueId id{0};
  Op op{Op::Write};
  unsigned char pattern{0};
};

// Two engines connected over loopback, one GPU buffer each. Children exit with
// _Exit, so nothing here is torn down: a faulted engine may not shut down cleanly,
// and that is not what these checks are about.
class LoopbackRig {
 public:
  LoopbackRig(bool notify, int qpPerTransfer) : notify_(notify) {
    if (!RdmaBackend::HasActiveDevices()) throw TestSkip("requires an active RDMA device");
    int gpus = 0;
    if (hipGetDeviceCount(&gpus) != hipSuccess || gpus == 0) throw TestSkip("requires a GPU");

    initiator_ = MakeEngine(kInitiatorKey, notify, qpPerTransfer);
    target_ = MakeEngine("fault_target", notify, qpPerTransfer);
    initiator_->RegisterRemoteEngine(target_->GetEngineDesc());
    target_->RegisterRemoteEngine(initiator_->GetEngineDesc());
    local_ = std::make_unique<GpuBuffer>(initiator_.get());
    remote_ = std::make_unique<GpuBuffer>(target_.get());
  }

  // Starts one transfer with a fresh data pattern in its source and zeros in its destination.
  Transfer* Start(Op op) {
    transfers_.push_back(std::make_unique<Transfer>());
    Transfer* t = transfers_.back().get();
    t->op = op;
    t->id = initiator_->AllocateTransferUniqueId();
    t->pattern = static_cast<unsigned char>(transfers_.size() % 250 + 1);
    GpuBuffer* src = op == Op::Read ? remote_.get() : local_.get();
    HIP_RUNTIME_CHECK(hipMemset(src->ptr, t->pattern, kBufBytes));
    HIP_RUNTIME_CHECK(hipMemset(Destination(op)->ptr, 0, kBufBytes));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    switch (op) {
      case Op::Write:
        initiator_->Write(local_->desc, 0, remote_->desc, 0, kBufBytes, &t->status, t->id);
        break;
      case Op::Read:
        initiator_->Read(local_->desc, 0, remote_->desc, 0, kBufBytes, &t->status, t->id);
        break;
      case Op::Batch: {
        MemDescVec local{local_->desc};
        MemDescVec remote{remote_->desc};
        BatchSizeVec offsets{{0, kSeg2Offset}};
        BatchSizeVec sizes{{kSegBytes, kSegBytes}};
        TransferStatusPtrVec statuses{&t->status};
        TransferUniqueIdVec ids{t->id};
        initiator_->BatchWrite(local, offsets, remote, offsets, sizes, statuses, ids);
        break;
      }
    }
    return t;
  }

  // The transfer's bytes are in its destination.
  bool DataLanded(const Transfer& t) {
    std::vector<unsigned char> host(kBufBytes);
    HIP_RUNTIME_CHECK(
        hipMemcpy(host.data(), Destination(t.op)->ptr, kBufBytes, hipMemcpyDeviceToHost));
    auto filled = [&](size_t off, size_t len) {
      for (size_t i = off; i < off + len; ++i) {
        if (host[i] != t.pattern) return false;
      }
      return true;
    };
    if (t.op == Op::Batch) return filled(0, kSegBytes) && filled(kSeg2Offset, kSegBytes);
    return filled(0, kBufBytes);
  }

  // With notifications off there is nothing to tell the target, so this holds.
  bool TargetTold(const Transfer& t, int timeoutMs) {
    if (!notify_) return true;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (true) {
      TransferStatus inbound;
      if (target_->PopInboundTransferStatus(kInitiatorKey, t.id, &inbound)) return true;
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // SUCCESS within the deadline, data landed, target told.
  bool Completes(Transfer* t, std::string* why) {
    StatusCode code = t->status.WaitFor(kDeadlineMs);
    if (code != StatusCode::SUCCESS) {
      *why = std::string(CodeName(code)) + " " + t->status.Message();
      return false;
    }
    if (!DataLanded(*t)) {
      *why = "SUCCESS but data did not land";
      return false;
    }
    if (!TargetTold(*t, kDeadlineMs)) {
      *why = "SUCCESS but the target was never notified";
      return false;
    }
    return true;
  }

 private:
  GpuBuffer* Destination(Op op) { return op == Op::Read ? local_.get() : remote_.get(); }

  static std::unique_ptr<IOEngine> MakeEngine(const std::string& key, bool notify,
                                              int qpPerTransfer) {
    IOEngineConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = GetFreePort();
    if (cfg.port <= 0) throw std::runtime_error("failed to allocate a tcp port");
    auto engine = std::make_unique<IOEngine>(key, cfg);
    RdmaBackendConfig rdmaCfg{};
    rdmaCfg.enableNotification = notify;
    rdmaCfg.notifPerQp = 64;
    rdmaCfg.qpPerTransfer = qpPerTransfer;
    engine->CreateBackend(BackendType::RDMA, rdmaCfg);
    return engine;
  }

  bool notify_;
  std::unique_ptr<IOEngine> initiator_;
  std::unique_ptr<IOEngine> target_;
  std::unique_ptr<GpuBuffer> local_;
  std::unique_ptr<GpuBuffer> remote_;
  std::vector<std::unique_ptr<Transfer>> transfers_;  // MORI keeps pointers to their statuses
};

/* ---------------------------------------------------------------------------------------------- */
/*                                     One entry (child process)                                  */
/* ---------------------------------------------------------------------------------------------- */

// Prints "RESULT ends=<0|1> honest=<0|1> recovers=<0|1>" for the parent to tabulate.
int RunEntry(const Entry& e) {
  LoopbackRig rig(e.notify, e.op == Op::Batch ? 2 : 1);
  std::string why;
  for (int i = 0; i < e.warmup; ++i) {
    if (!rig.Completes(rig.Start(e.op), &why)) {
      std::printf("    warmup transfer %d failed: %s\n", i, why.c_str());
      return kExitFail;
    }
  }

  const bool faulted = e.rule[0] != '\0';
  if (faulted) FaultInjector::Instance().Arm(ParseFaultRule(e.rule));

  Transfer* t = rig.Start(e.op);
  StatusCode code = t->status.WaitFor(kDeadlineMs);
  bool ends = code != StatusCode::IN_PROGRESS;
  bool honest = true;
  if (code == StatusCode::SUCCESS) {
    honest = rig.DataLanded(*t) && rig.TargetTold(*t, kDeadlineMs);
  }
  std::printf("    faulted transfer: %s %s\n", CodeName(code), t->status.Message().c_str());

  uint64_t fired = FaultInjector::Instance().Fired();
  FaultInjector::Instance().Disarm();
  if (faulted && fired == 0) {
    std::printf("    the fault never fired: the entry does not exercise its rule\n");
    return kExitFail;
  }

  bool recovers = rig.Completes(rig.Start(e.op), &why);
  if (!recovers) std::printf("    after the fault cleared: %s\n", why.c_str());

  std::printf("RESULT ends=%d honest=%d recovers=%d\n", ends, honest, recovers);
  return ends && honest && recovers ? kExitPass : kExitFail;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Runner (parent process)                                  */
/* ---------------------------------------------------------------------------------------------- */

struct Outcome {
  enum { Pass, Fail, Skip } verdict{Fail};
  // -1 = not reached (the child crashed or hung before reporting)
  int ends{-1}, honest{-1}, recovers{-1};
  bool alive{true};
  std::string note;
};

// Runs one entry in a child process, echoing its output, and reads back its result.
Outcome RunChild(const char* self, const Entry& e) {
  Outcome out;
  int fds[2];
  if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork failed");
  if (pid == 0) {
    dup2(fds[1], STDOUT_FILENO);
    close(fds[0]);
    close(fds[1]);
    execl(self, self, "--entry", e.id, static_cast<char*>(nullptr));
    _exit(127);
  }
  close(fds[1]);

  std::string text;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kChildTimeoutMs);
  bool timedOut = false;
  while (true) {
    pollfd p{fds[0], POLLIN, 0};
    if (poll(&p, 1, 100) > 0) {
      char buf[4096];
      ssize_t n = read(fds[0], buf, sizeof(buf));
      if (n <= 0) break;  // child closed stdout: exited
      text.append(buf, n);
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      timedOut = true;
      break;
    }
  }
  close(fds[0]);
  int wstatus = 0;
  waitpid(pid, &wstatus, 0);

  size_t start = 0;
  while (start < text.size()) {
    size_t end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(start, end - start);
    if (line.rfind("RESULT ", 0) == 0) {
      std::sscanf(line.c_str(), "RESULT ends=%d honest=%d recovers=%d", &out.ends, &out.honest,
                  &out.recovers);
    } else if (!line.empty()) {
      std::printf("%s\n", line.c_str());
    }
    start = end + 1;
  }

  if (timedOut) {
    out.alive = false;
    out.note = "killed after " + std::to_string(kChildTimeoutMs / 1000) + " s";
  } else if (WIFSIGNALED(wstatus)) {
    out.alive = false;
    out.note = std::string("crashed: ") + strsignal(WTERMSIG(wstatus));
  } else if (WEXITSTATUS(wstatus) == kExitSkip) {
    out.verdict = Outcome::Skip;
    return out;
  } else if (WEXITSTATUS(wstatus) != kExitPass && WEXITSTATUS(wstatus) != kExitFail) {
    out.alive = false;
    out.note = "process exited with code " + std::to_string(WEXITSTATUS(wstatus));
  }
  if (out.alive && WEXITSTATUS(wstatus) == kExitPass) out.verdict = Outcome::Pass;
  return out;
}

const char* Mark(int v) { return v < 0 ? "-" : v ? "ok" : "FAIL"; }

void CheckParser() {
  for (const Entry& e : kCatalog) {
    if (e.rule[0] != '\0') ParseFaultRule(e.rule);  // throws on a bad catalog rule
  }
  for (const char* bad : {"no_such_kind", "cqe_drop:skip", "cqe_drop:bogus=1", "qp_error:skip=x"}) {
    bool threw = false;
    try {
      ParseFaultRule(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) throw std::runtime_error(std::string("expected a parse error for '") + bad + "'");
  }
  FaultRule r = ParseFaultRule("cqe_drop:op=write:skip=3:count=2:qpn=17");
  if (r.kind != FaultKind::CqeDrop || r.wcOpcode != IBV_WC_RDMA_WRITE || r.skip != 3 ||
      r.count != 2 || r.qpn != 17) {
    throw std::runtime_error("cqe_drop spec parsed wrong");
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Engines re-read the IO level from the environment; keep the injector's
  // "armed"/"injected" warnings visible unless the caller chose a level.
  setenv("MORI_IO_LOG_LEVEL", "warn", /*overwrite=*/0);
  SetLogLevel("warn");

  if (argc == 3 && std::strcmp(argv[1], "--entry") == 0) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // keep what was printed if the entry crashes
    int rc = kExitFail;
    for (const Entry& e : kCatalog) {
      if (std::strcmp(e.id, argv[2]) != 0) continue;
      try {
        rc = RunEntry(e);
      } catch (const TestSkip& ex) {
        std::printf("    skipped: %s\n", ex.what());
        rc = kExitSkip;
      } catch (const std::exception& ex) {
        std::printf("    error: %s\n", ex.what());
      }
    }
    std::fflush(stdout);
    std::_Exit(rc);  // skip teardown: a faulted engine may never shut down
  }

  const char* filter = argc > 1 ? argv[1] : "";
  try {
    CheckParser();
    std::printf("[PASS] parser accepts every catalog rule and rejects malformed ones\n");
  } catch (const std::exception& ex) {
    std::printf("[FAIL] parser: %s\n", ex.what());
    return 1;
  }

  std::vector<std::pair<const Entry*, Outcome>> results;
  for (const Entry& e : kCatalog) {
    if (std::strstr(e.id, filter) == nullptr) continue;
    std::printf("---- %s: %s [%s]\n", e.id, e.realWorld, e.rule[0] ? e.rule : "no fault");
    std::fflush(stdout);
    results.emplace_back(&e, RunChild("/proc/self/exe", e));
  }

  int failed = 0, skipped = 0;
  std::printf("\n%-36s %-6s %-6s %-8s %-6s %s\n", "entry", "ends", "honest", "recovers", "alive",
              "verdict");
  for (const auto& [e, o] : results) {
    const char* verdict = o.verdict == Outcome::Pass   ? "PASS"
                          : o.verdict == Outcome::Skip ? "SKIP"
                                                       : "FAIL";
    failed += o.verdict == Outcome::Fail;
    skipped += o.verdict == Outcome::Skip;
    std::printf("%-36s %-6s %-6s %-8s %-6s %s%s%s\n", e->id, Mark(o.ends), Mark(o.honest),
                Mark(o.recovers), o.alive ? "ok" : "FAIL", verdict, o.note.empty() ? "" : "  ",
                o.note.c_str());
  }
  std::printf("==== test_fault_injection: %zu entries, %d failed, %d skipped ====\n",
              results.size(), failed, skipped);
  return failed == 0 ? 0 : 1;
}
