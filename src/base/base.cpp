#include "base/base.h"

#include <string>

namespace base {
// ============================================================================
// 1. Status 类的核心实现
// ============================================================================
// 构造函数：把传入的错误码和信息存储起来
Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

// 跨界赋值：把int值传给Status
Status& Status::operator=(int code) {
    code_ = code;
    return *this;  // 返回对象自身的引用
};

// 跨界判等
bool Status::operator==(int code) const {
    if (code_ == code) {
        return true;
    } else {
        return false;
    }
};

//跨界判异
bool Status::operator!=(int code) const {
    if (code_ != code) {
        return true;
    }
    else {
        return false;
    }
}

//伪装int
Status::operator int() const {
    return code_;
}

//伪装bool，如果code_是0（kSuccess）就返回true
Status::operator bool() const {
    return code_ == StatusCode::kSuccess;
}

int32_t Status::get_err_code() const {
    return code_;
}

const std::string& Status::get_err_msg() const {
    return message_;
}

void Status::set_err_msg(const std::string& err_msg) {
    message_ = err_msg;
}

// ============================================================================
// 2. error 命名空间下的快捷工厂函数
// ============================================================================
namespace error {
Status Success(const std::string& err_msg) {
    return Status(StatusCode::kSuccess, err_msg);
}

Status FunctionNotImplement(const std::string& err_msg) {
    return Status(StatusCode::kFunctionUnImplement, err_msg);
}

Status PathNotValid(const std::string& err_msg) {
    return Status(StatusCode::kPathNotValid, err_msg);
}

Status ModelParseError(const std::string& err_msg) {
    return Status(StatusCode::kModelParseError, err_msg);
}

Status InternalError(const std::string& err_msg) {
    return Status(StatusCode::kInternalError, err_msg);
}

Status KeyHasExits(const std::string& err_msg) {
    return Status(StatusCode::kKeyValueHasExist, err_msg);
}

Status InvalidArgument(const std::string& err_msg) {
    return Status(StatusCode::kInvalidArgument, err_msg);
}
}

// ============================================================================
// 3. 重载全局的输出流操作符 (operator<<)
// ============================================================================
// 让 std::cout << status 能够直接打印出好看的信息
std::ostream& operator<<(std::ostream& os, const Status& x) {
    os << x.get_err_msg();
    return os;
}

}


