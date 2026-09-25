#include "test_framework.hpp"

int main() {
    return kavach::test::TestRegistry::instance().run_all();
}
