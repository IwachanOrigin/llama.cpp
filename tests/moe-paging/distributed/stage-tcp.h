#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace stage_tcp {
using clock = std::chrono::steady_clock;
constexpr auto timeout = std::chrono::seconds(10);

#ifdef _WIN32
using handle = SOCKET;
constexpr handle invalid = INVALID_SOCKET;
inline int error() { return WSAGetLastError(); }
inline bool retry(int e) { return e == WSAEWOULDBLOCK || e == WSAEINTR; }
inline void close(handle fd) { closesocket(fd); }
struct runtime {
    runtime() {
        WSADATA info{};
        if (WSAStartup(MAKEWORD(2, 2), &info)) throw std::runtime_error("WSAStartup failed");
    }
    ~runtime() { WSACleanup(); }
};
#else
using handle = int;
constexpr handle invalid = -1;
inline int error() { return errno; }
inline bool retry(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
inline void close(handle fd) { ::close(fd); }
struct runtime {};
#endif

struct socket {
    handle fd = invalid;
    explicit socket(handle value = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) : fd(value) {
        if (fd == invalid) throw std::runtime_error("socket creation failed");
#ifdef _WIN32
        unsigned long enabled = 1;
        const bool ok = ioctlsocket(fd, FIONBIO, &enabled) == 0;
#else
        const bool ok = fd < FD_SETSIZE && fcntl(fd, F_SETFL, O_NONBLOCK) == 0;
#endif
        if (!ok) {
            close(fd);
            throw std::runtime_error("cannot enable nonblocking I/O");
        }
    }
    socket(const socket &) = delete;
    socket & operator=(const socket &) = delete;
    socket(socket && other) noexcept : fd(std::exchange(other.fd, invalid)) {}
    ~socket() { if (fd != invalid) close(fd); }

    void wait(bool writing, clock::time_point deadline) const {
        for (;;) {
            const auto left = std::chrono::duration_cast<std::chrono::microseconds>(deadline - clock::now()).count();
            if (left <= 0) throw std::runtime_error("TCP timeout");
            timeval tv{long(left / 1000000), long(left % 1000000)};
            fd_set want, exceptions;
            FD_ZERO(&want);
            FD_ZERO(&exceptions);
            FD_SET(fd, &want);
            FD_SET(fd, &exceptions);
#ifdef _WIN32
            const int width = 0;
#else
            const int width = fd + 1;
#endif
            const int result = select(width, writing ? nullptr : &want, writing ? &want : nullptr, &exceptions, &tv);
            if (result > 0) return;
            if (result == 0) throw std::runtime_error("TCP timeout");
            if (!retry(error())) throw std::runtime_error("TCP readiness check failed");
        }
    }

    void transfer(uint8_t * bytes, size_t length, bool writing, clock::time_point deadline, uint64_t & total) {
        while (length) {
            wait(writing, deadline);
            const int chunk = int(length > 65536 ? 65536 : length);
#ifdef _WIN32
            constexpr int flags = 0;
#else
            constexpr int flags = MSG_NOSIGNAL;
#endif
            const int moved = writing ? ::send(fd, reinterpret_cast<const char *>(bytes), chunk, flags)
                                      : ::recv(fd, reinterpret_cast<char *>(bytes), chunk, 0);
            if (moved == 0) throw std::runtime_error("peer closed during frame");
            if (moved < 0) {
                if (retry(error())) continue;
                throw std::runtime_error("TCP transfer failed");
            }
            total += moved;
            bytes += moved;
            length -= moved;
        }
    }
};

inline sockaddr_in address(uint16_t port) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    return addr;
}

inline socket connect(uint16_t port, std::chrono::milliseconds limit) {
    socket peer;
    const auto addr = address(port);
    if (::connect(peer.fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0) {
        const auto code = error();
#ifdef _WIN32
        if (!retry(code)) throw std::runtime_error("TCP connect failed");
#else
        if (code != EINPROGRESS && !retry(code)) throw std::runtime_error("TCP connect failed");
#endif
        peer.wait(true, clock::now() + limit);
        int status = 0;
#ifdef _WIN32
        int size = sizeof(status);
#else
        socklen_t size = sizeof(status);
#endif
        if (getsockopt(peer.fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&status), &size) || status) {
            throw std::runtime_error("TCP connection refused or failed");
        }
    }
    return peer;
}

inline uint16_t listen(socket & listener, uint16_t port) {
    auto addr = address(port);
#ifdef _WIN32
    int exclusive = 1;
    if (setsockopt(listener.fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
            reinterpret_cast<const char *>(&exclusive), sizeof(exclusive))) {
        throw std::runtime_error("cannot set exclusive bind");
    }
    int size = sizeof(addr);
#else
    socklen_t size = sizeof(addr);
#endif
    if (::bind(listener.fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) ||
        ::listen(listener.fd, 1) || getsockname(listener.fd, reinterpret_cast<sockaddr *>(&addr), &size)) {
        throw std::runtime_error("loopback bind/listen failed");
    }
    return ntohs(addr.sin_port);
}

inline socket accept(socket & listener, std::chrono::milliseconds limit) {
    const auto deadline = clock::now() + limit;
    for (;;) {
        listener.wait(false, deadline);
        const auto fd = ::accept(listener.fd, nullptr, nullptr);
        if (fd != invalid) return socket(fd);
        if (!retry(error())) throw std::runtime_error("TCP accept failed");
    }
}
} // namespace stage_tcp
