#pragma once

#include <string>

namespace p10 {
class Tensor;
}  // namespace p10

namespace p10::tlog {

/// Sends `tensor` to the tensor log server under the name `entry`.
///
/// The first call opens the connection, to `PTENSOR_TLOG_ADDRESS` when that
/// variable is set and to "localhost:4449" otherwise. Failures are logged, not
/// reported: logging is a debugging aid and must not change the caller's flow.
void log(const std::string& entry, const Tensor& tensor);

/// Same, to an explicit "host:port" rather than the process wide endpoint.
///
/// Each address keeps its own session, opened on first use and held until the
/// process exits, so repeated calls to one address reuse a connection. Meant
/// for callers that pick the endpoint at run time -- a debugger evaluating
/// `p10::tlog::log_to("127.0.0.1:<port>", "x", x)` to push a tensor into a
/// viewer listening on a port only the viewer knows.
///
/// Takes `const char*`, not `std::string`: an expression evaluator binds a
/// string literal to it directly, while a `const std::string&` parameter needs
/// a conversion the debugger will not do for you.
void log_to(const char* address, const char* entry, const Tensor& tensor);

}  // namespace p10::tlog
