#include <dinoml/abi.h>

#include <string>

static thread_local std::string g_last_error;

extern "C" {

int dino_runtime_fail(const char* message) {
  g_last_error = message == nullptr ? "Unknown DinoML runtime error" : message;
  return 1;
}

int dino_abi_version() {
  return DINO_RUNTIME_ABI_VERSION;
}

const char* dino_get_last_error() {
  return g_last_error.c_str();
}

}
