#include <cmath>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "../test_framework.hpp"
#include "core/error.hpp"
#include "io/parser.hpp"
#include "model/model.hpp"

using namespace kavach;
using namespace kavach::io;
using namespace kavach::model;

namespace {

[[nodiscard]] std::filesystem::path find_netlib_dir() {
    std::vector<std::filesystem::path> candidates = {"benchmarks/netlib", "../benchmarks/netlib",
                                                     "../../benchmarks/netlib"};
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p) && std::filesystem::exists(p / "afiro.mps")) {
            return p;
        }
    }
    throw std::runtime_error("Could not find benchmarks/netlib directory containing afiro.mps");
}

}  // namespace

// ============================================================================
// 1. MPS Parser Unit Tests
// ============================================================================

KAVACH_TEST(test_mps_parser_basic_lp) {
    constexpr std::string_view mps_content = R"(
NAME          TESTPROD
ROWS
 N  COST
 L  CAPACITY
 G  DEMAND
 E  EQUALITY
COLUMNS
    X1        COST              1.5   CAPACITY          2.0
    X1        DEMAND            1.0
    X2        COST              2.5   CAPACITY          1.0
    X2        EQUALITY          3.0
RHS
    RHS1      CAPACITY         10.0   DEMAND            5.0
    RHS1      EQUALITY          6.0
ENDATA
)";

    MpsParser parser;
    auto model = parser.parse_string(mps_content, "testprod.mps");

    KAVACH_ASSERT(model != nullptr);
    KAVACH_ASSERT_EQUAL(model->name(), "TESTPROD");
    KAVACH_ASSERT_EQUAL(model->num_variables(), 2u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 3u);

    // Variables X1, X2 default to [0, +inf) continuous
    auto x1_opt = model->find_variable_id("X1");
    auto x2_opt = model->find_variable_id("X2");
    KAVACH_ASSERT(x1_opt.has_value());
    KAVACH_ASSERT(x2_opt.has_value());

    const auto& x1 = model->variable(*x1_opt);
    const auto& x2 = model->variable(*x2_opt);
    KAVACH_ASSERT(x1.is_continuous());
    KAVACH_ASSERT_EQUAL(x1.lower_bound, 0.0);
    KAVACH_ASSERT(std::isinf(x1.upper_bound));
    KAVACH_ASSERT(x2.is_continuous());
    KAVACH_ASSERT_EQUAL(x2.lower_bound, 0.0);
    KAVACH_ASSERT(std::isinf(x2.upper_bound));

    // Objective COST: 1.5 X1 + 2.5 X2
    KAVACH_ASSERT_EQUAL(model->objective_sense(), ObjectiveSense::MINIMIZE);
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(*x1_opt), 1.5, 1e-9);
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(*x2_opt), 2.5, 1e-9);

    // Constraints
    auto cap_opt = model->find_constraint_id("CAPACITY");
    auto dem_opt = model->find_constraint_id("DEMAND");
    auto eq_opt = model->find_constraint_id("EQUALITY");
    KAVACH_ASSERT(cap_opt.has_value());
    KAVACH_ASSERT(dem_opt.has_value());
    KAVACH_ASSERT(eq_opt.has_value());

    const auto& cap = model->constraint(*cap_opt);
    KAVACH_ASSERT(cap.sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(cap.rhs, 10.0, 1e-9);
    KAVACH_ASSERT_NEAR(cap.expression.get_coefficient(*x1_opt), 2.0, 1e-9);
    KAVACH_ASSERT_NEAR(cap.expression.get_coefficient(*x2_opt), 1.0, 1e-9);

    const auto& dem = model->constraint(*dem_opt);
    KAVACH_ASSERT(dem.sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(dem.rhs, 5.0, 1e-9);

    const auto& eq = model->constraint(*eq_opt);
    KAVACH_ASSERT(eq.sense == ConstraintSense::EQUAL);
    KAVACH_ASSERT_NEAR(eq.rhs, 6.0, 1e-9);
}

KAVACH_TEST(test_mps_parser_all_bound_types) {
    constexpr std::string_view mps_content = R"(
NAME          BOUNDTEST
ROWS
 N  OBJ
 L  CON1
COLUMNS
    X_LO      OBJ               1.0   CON1              1.0
    X_UP      OBJ               1.0   CON1              1.0
    X_FX      OBJ               1.0   CON1              1.0
    X_FR      OBJ               1.0   CON1              1.0
    X_BV      OBJ               1.0   CON1              1.0
    X_LI      OBJ               1.0   CON1              1.0
    X_UI      OBJ               1.0   CON1              1.0
    X_MI      OBJ               1.0   CON1              1.0
    X_PL      OBJ               1.0   CON1              1.0
RHS
    RHS1      CON1            100.0
BOUNDS
 LO BND1      X_LO              2.5
 UP BND1      X_UP              8.0
 FX BND1      X_FX              3.14
 FR BND1      X_FR
 BV BND1      X_BV
 LI BND1      X_LI              5.0
 UI BND1      X_UI             15.0
 MI BND1      X_MI
 PL BND1      X_PL
ENDATA
)";

    MpsParser parser;
    auto model = parser.parse_string(mps_content, "boundtest.mps");

    const auto& v_lo = model->variable(*model->find_variable_id("X_LO"));
    KAVACH_ASSERT_NEAR(v_lo.lower_bound, 2.5, 1e-9);
    KAVACH_ASSERT(std::isinf(v_lo.upper_bound));

    const auto& v_up = model->variable(*model->find_variable_id("X_UP"));
    KAVACH_ASSERT_NEAR(v_up.lower_bound, 0.0, 1e-9);
    KAVACH_ASSERT_NEAR(v_up.upper_bound, 8.0, 1e-9);

    const auto& v_fx = model->variable(*model->find_variable_id("X_FX"));
    KAVACH_ASSERT_NEAR(v_fx.lower_bound, 3.14, 1e-9);
    KAVACH_ASSERT_NEAR(v_fx.upper_bound, 3.14, 1e-9);

    const auto& v_fr = model->variable(*model->find_variable_id("X_FR"));
    KAVACH_ASSERT(v_fr.is_free());

    const auto& v_bv = model->variable(*model->find_variable_id("X_BV"));
    KAVACH_ASSERT(v_bv.is_binary());
    KAVACH_ASSERT_NEAR(v_bv.lower_bound, 0.0, 1e-9);
    KAVACH_ASSERT_NEAR(v_bv.upper_bound, 1.0, 1e-9);

    const auto& v_li = model->variable(*model->find_variable_id("X_LI"));
    KAVACH_ASSERT(v_li.is_integer());
    KAVACH_ASSERT_NEAR(v_li.lower_bound, 5.0, 1e-9);

    const auto& v_ui = model->variable(*model->find_variable_id("X_UI"));
    KAVACH_ASSERT(v_ui.is_integer());
    KAVACH_ASSERT_NEAR(v_ui.upper_bound, 15.0, 1e-9);

    const auto& v_mi = model->variable(*model->find_variable_id("X_MI"));
    KAVACH_ASSERT(std::isinf(v_mi.lower_bound) && v_mi.lower_bound < 0.0);

    const auto& v_pl = model->variable(*model->find_variable_id("X_PL"));
    KAVACH_ASSERT(std::isinf(v_pl.upper_bound) && v_pl.upper_bound > 0.0);
}

KAVACH_TEST(test_mps_parser_integer_markers) {
    constexpr std::string_view mps_content = R"(
NAME          INTTEST
ROWS
 N  OBJ
 L  CON1
COLUMNS
    X_CONT1   OBJ               1.0   CON1              1.0
    MARK01    'MARKER'                 'INTORG'
    X_INT1    OBJ               2.0   CON1              2.0
    X_INT2    OBJ               3.0   CON1              3.0
    MARK02    'MARKER'                 'INTEND'
    X_CONT2   OBJ               4.0   CON1              4.0
RHS
    R1        CON1             50.0
ENDATA
)";

    MpsParser parser;
    auto model = parser.parse_string(mps_content, "inttest.mps");

    KAVACH_ASSERT_EQUAL(model->num_variables(), 4u);
    const auto& c1 = model->variable(*model->find_variable_id("X_CONT1"));
    const auto& i1 = model->variable(*model->find_variable_id("X_INT1"));
    const auto& i2 = model->variable(*model->find_variable_id("X_INT2"));
    const auto& c2 = model->variable(*model->find_variable_id("X_CONT2"));

    KAVACH_ASSERT(c1.is_continuous());
    KAVACH_ASSERT(i1.is_integer());
    KAVACH_ASSERT(i2.is_integer());
    KAVACH_ASSERT(c2.is_continuous());
    KAVACH_ASSERT(model->is_milp());
}

KAVACH_TEST(test_mps_parser_ranges_dual_expansion) {
    constexpr std::string_view mps_content = R"(
NAME          RANGETEST
ROWS
 N  OBJ
 G  ROW_G
 L  ROW_L
 E  ROW_EPOS
 E  ROW_ENEG
COLUMNS
    X1        OBJ               1.0   ROW_G             1.0
    X1        ROW_L             1.0   ROW_EPOS          1.0
    X1        ROW_ENEG          1.0
RHS
    RHS1      ROW_G            10.0   ROW_L            20.0
    RHS1      ROW_EPOS         30.0   ROW_ENEG         40.0
RANGES
    RNG1      ROW_G             5.0   ROW_L             4.0
    RNG1      ROW_EPOS          6.0   ROW_ENEG         -7.0
ENDATA
)";

    MpsParser parser;
    auto model = parser.parse_string(mps_content, "ranges.mps");

    // 4 ranged rows should produce 8 constraints:
    // ROW_G (>= 10) and ROW_G_range_ub (<= 15)
    // ROW_L (<= 20) and ROW_L_range_lb (>= 16)
    // ROW_EPOS_range_lb (>= 30) and ROW_EPOS_range_ub (<= 36)
    // ROW_ENEG_range_lb (>= 33) and ROW_ENEG_range_ub (<= 40)
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 8u);

    auto g1 = model->find_constraint_id("ROW_G");
    auto g2 = model->find_constraint_id("ROW_G_range_ub");
    KAVACH_ASSERT(g1.has_value() && g2.has_value());
    KAVACH_ASSERT(model->constraint(*g1).sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*g1).rhs, 10.0, 1e-9);
    KAVACH_ASSERT(model->constraint(*g2).sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*g2).rhs, 15.0, 1e-9);

    auto l1 = model->find_constraint_id("ROW_L");
    auto l2 = model->find_constraint_id("ROW_L_range_lb");
    KAVACH_ASSERT(l1.has_value() && l2.has_value());
    KAVACH_ASSERT(model->constraint(*l1).sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*l1).rhs, 20.0, 1e-9);
    KAVACH_ASSERT(model->constraint(*l2).sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*l2).rhs, 16.0, 1e-9);

    auto ep_lb = model->find_constraint_id("ROW_EPOS_range_lb");
    auto ep_ub = model->find_constraint_id("ROW_EPOS_range_ub");
    KAVACH_ASSERT(ep_lb.has_value() && ep_ub.has_value());
    KAVACH_ASSERT(model->constraint(*ep_lb).sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*ep_lb).rhs, 30.0, 1e-9);
    KAVACH_ASSERT(model->constraint(*ep_ub).sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*ep_ub).rhs, 36.0, 1e-9);

    auto en_lb = model->find_constraint_id("ROW_ENEG_range_lb");
    auto en_ub = model->find_constraint_id("ROW_ENEG_range_ub");
    KAVACH_ASSERT(en_lb.has_value() && en_ub.has_value());
    KAVACH_ASSERT(model->constraint(*en_lb).sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*en_lb).rhs, 33.0, 1e-9);
    KAVACH_ASSERT(model->constraint(*en_ub).sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*en_ub).rhs, 40.0, 1e-9);
}

KAVACH_TEST(test_mps_parser_free_rows_and_obj_selection) {
    constexpr std::string_view mps_content = R"(
NAME          FREEROWS
ROWS
 N  FREE_ROW1
 N  REAL_OBJ
 N  FREE_ROW2
 L  CONSTR1
COLUMNS
    X1        FREE_ROW1         10.0   REAL_OBJ          2.0
    X1        FREE_ROW2         20.0   CONSTR1           1.0
RHS
    RHS1      CONSTR1            5.0
ENDATA
)";

    ParserOptions opts;
    opts.objective_name = "REAL_OBJ";

    MpsParser parser;
    auto model = parser.parse_string(mps_content, "freerows.mps", opts);

    // Exactly 1 constraint (CONSTR1). The free rows must NOT be added as constraints!
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 1u);
    KAVACH_ASSERT_EQUAL(model->constraint(0).name, "CONSTR1");

    auto x1_id = *model->find_variable_id("X1");
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x1_id), 2.0, 1e-9);
}

KAVACH_TEST(test_mps_parser_number_dialects) {
    constexpr std::string_view mps_content = R"(
NAME          NUMTEST
ROWS
 N  OBJ
 L  CON1
COLUMNS
    X1        OBJ               .301   CON1               310.
    X2        OBJ            1.25D-02  CON1            2.50+02
RHS
    R1        CON1               -1.
ENDATA
)";

    MpsParser parser;
    auto model = parser.parse_string(mps_content, "numtest.mps");

    auto x1_id = *model->find_variable_id("X1");
    auto x2_id = *model->find_variable_id("X2");

    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x1_id), 0.301, 1e-9);
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x2_id), 0.0125, 1e-9);

    const auto& con = model->constraint(0);
    KAVACH_ASSERT_NEAR(con.rhs, -1.0, 1e-9);
    KAVACH_ASSERT_NEAR(con.expression.get_coefficient(x1_id), 310.0, 1e-9);
    KAVACH_ASSERT_NEAR(con.expression.get_coefficient(x2_id), 250.0, 1e-9);
}

// ============================================================================
// 2. Native LP Parser Unit Tests
// ============================================================================

KAVACH_TEST(test_lp_parser_basic_linear) {
    constexpr std::string_view lp_content = R"(
\ Simple LP test model
Maximize
 obj: 10 x1 + 6 x2 + 4 x3
Subject To
 c1: x1 + x2 + x3 <= 100
 c2: 10 x1 + 4 x2 + 5 x3 <= 600
 c3: 2 x1 + 2 x2 + 6 x3 <= 300
Bounds
 0 <= x1 <= 50
 x2 >= 10
 x3 free
End
)";

    LpParser parser;
    auto model = parser.parse_string(lp_content, "simple.lp");

    KAVACH_ASSERT_EQUAL(model->objective_sense(), ObjectiveSense::MAXIMIZE);
    KAVACH_ASSERT_EQUAL(model->num_variables(), 3u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 3u);

    auto x1_id = *model->find_variable_id("x1");
    auto x2_id = *model->find_variable_id("x2");
    auto x3_id = *model->find_variable_id("x3");

    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x1_id), 10.0, 1e-9);
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x2_id), 6.0, 1e-9);
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x3_id), 4.0, 1e-9);

    const auto& v1 = model->variable(x1_id);
    KAVACH_ASSERT_NEAR(v1.lower_bound, 0.0, 1e-9);
    KAVACH_ASSERT_NEAR(v1.upper_bound, 50.0, 1e-9);

    const auto& v2 = model->variable(x2_id);
    KAVACH_ASSERT_NEAR(v2.lower_bound, 10.0, 1e-9);
    KAVACH_ASSERT(std::isinf(v2.upper_bound));

    const auto& v3 = model->variable(x3_id);
    KAVACH_ASSERT(v3.is_free());
}

KAVACH_TEST(test_lp_parser_quadratic_objective) {
    constexpr std::string_view lp_content = R"(
Minimize
 cost: 3 x1 - 2 x2 + [ 4 x1 ^ 2 + 6 x1 * x2 + 8 x2 ^ 2 ] / 2
Subject To
 limit: x1 + x2 >= 5
Bounds
 x1 >= 0
 x2 >= 0
End
)";

    LpParser parser;
    auto model = parser.parse_string(lp_content, "qp.lp");

    KAVACH_ASSERT(model->is_qp());
    KAVACH_ASSERT_EQUAL(model->objective().quadratic_terms.num_terms(), 3u);

    auto x1_id = *model->find_variable_id("x1");
    auto x2_id = *model->find_variable_id("x2");

    // [ 4 x1^2 + 6 x1*x2 + 8 x2^2 ] / 2 -> 2 x1^2, 3 x1*x2, 4 x2^2
    for (const auto& term : model->objective().quadratic_terms.terms()) {
        if (term.var1_id == x1_id && term.var2_id == x1_id) {
            KAVACH_ASSERT_NEAR(term.coefficient, 2.0, 1e-9);
        } else if ((term.var1_id == x1_id && term.var2_id == x2_id) ||
                   (term.var1_id == x2_id && term.var2_id == x1_id)) {
            KAVACH_ASSERT_NEAR(term.coefficient, 3.0, 1e-9);
        } else if (term.var1_id == x2_id && term.var2_id == x2_id) {
            KAVACH_ASSERT_NEAR(term.coefficient, 4.0, 1e-9);
        }
    }
}

KAVACH_TEST(test_lp_parser_range_constraints) {
    constexpr std::string_view lp_content = R"(
Minimize
 x1 + x2
Subject To
 range_c: 10 <= 2 x1 + 3 x2 <= 50
Bounds
 x1 >= 0
 x2 >= 0
End
)";

    LpParser parser;
    auto model = parser.parse_string(lp_content, "range.lp");

    // Dual constraints generated for two-sided inequality
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 2u);
    auto lb_id = model->find_constraint_id("range_c_lb");
    auto ub_id = model->find_constraint_id("range_c_ub");
    KAVACH_ASSERT(lb_id.has_value() && ub_id.has_value());

    KAVACH_ASSERT(model->constraint(*lb_id).sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*lb_id).rhs, 10.0, 1e-9);

    KAVACH_ASSERT(model->constraint(*ub_id).sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*ub_id).rhs, 50.0, 1e-9);
}

KAVACH_TEST(test_lp_parser_bounds_and_binaries) {
    constexpr std::string_view lp_content = R"(
Minimize
 5 x + 3 y + 2 z + 8 w
Subject To
 x + y + z + w >= 1
Bounds
 w = 4.5
Generals
 x
 y
Binaries
 z
End
)";

    LpParser parser;
    auto model = parser.parse_string(lp_content, "milp.lp");

    KAVACH_ASSERT(model->is_milp());
    const auto& x = model->variable(*model->find_variable_id("x"));
    const auto& y = model->variable(*model->find_variable_id("y"));
    const auto& z = model->variable(*model->find_variable_id("z"));
    const auto& w = model->variable(*model->find_variable_id("w"));

    KAVACH_ASSERT(x.is_integer());
    KAVACH_ASSERT(y.is_integer());
    KAVACH_ASSERT(z.is_binary());
    KAVACH_ASSERT_NEAR(z.lower_bound, 0.0, 1e-9);
    KAVACH_ASSERT_NEAR(z.upper_bound, 1.0, 1e-9);

    KAVACH_ASSERT(w.is_continuous());
    KAVACH_ASSERT_NEAR(w.lower_bound, 4.5, 1e-9);
    KAVACH_ASSERT_NEAR(w.upper_bound, 4.5, 1e-9);
}

// ============================================================================
// 3. Adversarial Malformed-File Tests
// ============================================================================

KAVACH_TEST(test_parser_adversarial_mps) {
    MpsParser parser;

    // A: Unknown section header
    constexpr std::string_view bad_section = "NAME BAD\nFOOBAR\nENDATA\n";
    bool caught1 = false;
    try {
        auto m = parser.parse_string(bad_section, "bad_sec.mps");
        (void)m;
    } catch (const ParserException& e) {
        caught1 = true;
        KAVACH_ASSERT_EQUAL(e.line(), 2u);
    }
    KAVACH_ASSERT(caught1);

    // B: Unknown row sense
    constexpr std::string_view bad_sense = "NAME BAD\nROWS\n Z ROW1\nCOLUMNS\nENDATA\n";
    bool caught2 = false;
    try {
        auto m = parser.parse_string(bad_sense, "bad_sense.mps");
        (void)m;
    } catch (const ParserException& e) {
        caught2 = true;
        KAVACH_ASSERT_EQUAL(e.section(), "ROWS");
    }
    KAVACH_ASSERT(caught2);

    // C: Duplicate row name
    constexpr std::string_view dup_rows =
        "NAME DUP\nROWS\n N OBJ\n L ROW1\n L ROW1\nCOLUMNS\nENDATA\n";
    bool caught3 = false;
    try {
        auto m = parser.parse_string(dup_rows, "dup_rows.mps");
        (void)m;
    } catch (const ParserException& e) {
        caught3 = true;
        KAVACH_ASSERT_EQUAL(e.section(), "ROWS");
    }
    KAVACH_ASSERT(caught3);

    // D: Undeclared row in COLUMNS
    constexpr std::string_view undeclared_row =
        "NAME UNDEC\nROWS\n N OBJ\nCOLUMNS\n    X1 UNDECLARED 1.0\nENDATA\n";
    bool caught4 = false;
    try {
        auto m = parser.parse_string(undeclared_row, "undec.mps");
        (void)m;
    } catch (const ParserException& e) {
        caught4 = true;
        KAVACH_ASSERT_EQUAL(e.section(), "COLUMNS");
    }
    KAVACH_ASSERT(caught4);

    // E: Undeclared column in BOUNDS
    constexpr std::string_view undeclared_col =
        "NAME UNDEC\nROWS\n N OBJ\nCOLUMNS\n    X1 OBJ 1.0\nBOUNDS\n UP BND X2 5.0\nENDATA\n";
    bool caught5 = false;
    try {
        auto m = parser.parse_string(undeclared_col, "undec_col.mps");
        (void)m;
    } catch (const ParserException& e) {
        caught5 = true;
        KAVACH_ASSERT_EQUAL(e.section(), "BOUNDS");
    }
    KAVACH_ASSERT(caught5);

    // F: NaN coefficient in COLUMNS
    constexpr std::string_view nan_coeff =
        "NAME NAN_TEST\nROWS\n N OBJ\nCOLUMNS\n    X1 OBJ nan\nENDATA\n";
    bool caught6 = false;
    try {
        auto m = parser.parse_string(nan_coeff, "nan.mps");
        (void)m;
    } catch (const ParserException& e) {
        caught6 = true;
    }
    KAVACH_ASSERT(caught6);
}

KAVACH_TEST(test_parser_adversarial_lp) {
    LpParser parser;

    // A: Missing objective keyword
    constexpr std::string_view missing_obj = "Subject To\n c1: x <= 10\nEnd\n";
    bool caught1 = false;
    try {
        auto m = parser.parse_string(missing_obj, "no_obj.lp");
        (void)m;
    } catch (const ParserException& e) {
        caught1 = true;
    }
    KAVACH_ASSERT(caught1);

    // B: Missing relational operator in constraint
    constexpr std::string_view missing_op = "Minimize\n x\nSubject To\n c1: 2 x 10\nEnd\n";
    bool caught2 = false;
    try {
        auto m = parser.parse_string(missing_op, "missing_op.lp");
        (void)m;
    } catch (const ParserException& e) {
        caught2 = true;
    }
    KAVACH_ASSERT(caught2);

    // C: Unclosed quadratic bracket
    constexpr std::string_view unclosed_bracket =
        "Minimize\n [ 2 x^2 + 3 y\nSubject To\n c1: x + y <= 1\nEnd\n";
    bool caught3 = false;
    try {
        auto m = parser.parse_string(unclosed_bracket, "unclosed.lp");
        (void)m;
    } catch (const ParserException& e) {
        caught3 = true;
    }
    KAVACH_ASSERT(caught3);
}

// ============================================================================
// 4. Authentic Netlib Benchmark Corpus Verification
// ============================================================================

KAVACH_TEST(test_netlib_corpus_afiro) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "afiro.mps");

    // Official Netlib AFIRO: 28 rows (incl obj), 32 cols, 88 nonzeros
    KAVACH_ASSERT_EQUAL(model->name(), "AFIRO");
    KAVACH_ASSERT_EQUAL(model->num_variables(), 32u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 27u);

    // Spot-checked coefficients
    auto x01_id = *model->find_variable_id("X01");
    auto x02_id = *model->find_variable_id("X02");
    auto r09_id = *model->find_constraint_id("R09");
    auto x48_id = *model->find_constraint_id("X48");

    KAVACH_ASSERT_NEAR(model->constraint(r09_id).expression.get_coefficient(x01_id), -1.0, 1e-9);
    KAVACH_ASSERT_NEAR(model->constraint(x48_id).expression.get_coefficient(x01_id), 0.301, 1e-9);
    KAVACH_ASSERT_NEAR(model->objective().linear_terms.get_coefficient(x02_id), -0.4, 1e-9);

    // RHS on constraint X50 is 310.0
    auto x50_id = *model->find_constraint_id("X50");
    KAVACH_ASSERT_NEAR(model->constraint(x50_id).rhs, 310.0, 1e-9);

    // Total nonzeros: constraints nonzeros + objective nonzeros = 88
    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 88u);
}

KAVACH_TEST(test_netlib_corpus_adlittle) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "adlittle.mps");

    // Official Netlib ADLITTLE: 57 rows (incl obj), 97 cols, 465 nonzeros
    KAVACH_ASSERT_EQUAL(model->name(), "ADLITTLE");
    KAVACH_ASSERT_EQUAL(model->num_variables(), 97u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 56u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 465u);
}

KAVACH_TEST(test_netlib_corpus_blend) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "blend.mps");

    // Official Netlib BLEND: 75 rows (incl obj), 83 cols, 521 nonzeros
    KAVACH_ASSERT_EQUAL(model->num_variables(), 83u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 74u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 521u);
}

KAVACH_TEST(test_netlib_corpus_sc50a_sc50b) {
    const auto netlib_dir = find_netlib_dir();

    // SC50A: 51 rows (incl obj), 48 cols
    auto m_sc50a = parse_model(netlib_dir / "sc50a.mps");
    KAVACH_ASSERT_EQUAL(m_sc50a->num_variables(), 48u);
    KAVACH_ASSERT_EQUAL(m_sc50a->num_constraints(), 50u);

    // SC50B: 51 rows (incl obj), 48 cols
    auto m_sc50b = parse_model(netlib_dir / "sc50b.mps");
    KAVACH_ASSERT_EQUAL(m_sc50b->num_variables(), 48u);
    KAVACH_ASSERT_EQUAL(m_sc50b->num_constraints(), 50u);
}

KAVACH_TEST(test_netlib_corpus_share2b) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "share2b.mps");

    // Official Netlib SHARE2B: 97 rows (incl obj), 79 cols, 730 nonzeros
    KAVACH_ASSERT_EQUAL(model->name(), "SHARE2B");
    KAVACH_ASSERT_EQUAL(model->num_variables(), 79u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 96u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 730u);
}

KAVACH_TEST(test_netlib_corpus_beaconfd) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "beaconfd.mps");

    // Official Netlib BEACONFD: 174 rows (incl obj), 262 cols, 3476 nonzeros
    KAVACH_ASSERT_EQUAL(model->num_variables(), 262u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 173u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 3476u);
}

KAVACH_TEST(test_netlib_corpus_sc105) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "sc105.mps");

    // Official Netlib SC105: 106 rows (incl obj), 103 cols
    KAVACH_ASSERT_EQUAL(model->num_variables(), 103u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 105u);
}

KAVACH_TEST(test_netlib_corpus_lotfi) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "lotfi.mps");

    // Official Netlib LOTFI: 154 rows (incl obj), 308 cols, 1086 nonzeros
    KAVACH_ASSERT_EQUAL(model->num_variables(), 308u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 153u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 1086u);
}

KAVACH_TEST(test_netlib_corpus_share1b) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "share1b.mps");

    // Official Netlib SHARE1B: 118 rows (incl obj), 225 cols, 1182 nonzeros
    KAVACH_ASSERT_EQUAL(model->num_variables(), 225u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 117u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 1182u);
}

KAVACH_TEST(test_netlib_corpus_kb2) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "kb2.mps");

    // Official Netlib KB2: 44 rows (incl obj), 41 cols, 291 nonzeros, BOUNDS
    KAVACH_ASSERT_EQUAL(model->name(), "KB2");
    KAVACH_ASSERT_EQUAL(model->num_variables(), 41u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 43u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 291u);

    // Spot-check bound on BHC.3EBW: UP 10.0
    auto bhc_id = *model->find_variable_id("BHC.3EBW");
    KAVACH_ASSERT_NEAR(model->variable(bhc_id).upper_bound, 10.0, 1e-9);
}

KAVACH_TEST(test_netlib_corpus_stocfor1) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "stocfor1.mps");

    // Official Netlib STOCFOR1: 118 rows (incl obj), 111 cols, 474 nonzeros
    KAVACH_ASSERT_EQUAL(model->num_variables(), 111u);
    KAVACH_ASSERT_EQUAL(model->num_constraints(), 117u);

    std::size_t total_nnz = model->num_nonzeros() + model->objective().linear_terms.num_terms();
    KAVACH_ASSERT_EQUAL(total_nnz, 474u);
}

KAVACH_TEST(test_netlib_corpus_boeing2) {
    const auto netlib_dir = find_netlib_dir();
    auto model = parse_model(netlib_dir / "boeing2.mps");

    // Official Netlib BOEING2: 143 cols. Contains RANGES and BOUNDS.
    KAVACH_ASSERT_EQUAL(model->name(), "BOEING2");
    KAVACH_ASSERT_EQUAL(model->num_variables(), 143u);

    // Spot check bound on GRDTIMN1: LO -100.0, UP 0.0
    auto v_id = *model->find_variable_id("GRDTIMN1");
    const auto& var = model->variable(v_id);
    KAVACH_ASSERT_NEAR(var.lower_bound, -100.0, 1e-9);
    KAVACH_ASSERT_NEAR(var.upper_bound, 0.0, 1e-9);

    // Verify dual constraints were expanded from RANGES:
    // DMBOSORD is an L row (<= 302.0) with range 61.0, generating DMBOSORD (<= 302) and
    // DMBOSORD_range_lb (>= 241)
    auto c_orig = model->find_constraint_id("DMBOSORD");
    auto c_dual = model->find_constraint_id("DMBOSORD_range_lb");
    KAVACH_ASSERT(c_orig.has_value());
    KAVACH_ASSERT(c_dual.has_value());
    KAVACH_ASSERT(model->constraint(*c_orig).sense == ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*c_orig).rhs, 302.0, 1e-9);
    KAVACH_ASSERT(model->constraint(*c_dual).sense == ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_NEAR(model->constraint(*c_dual).rhs, 241.0, 1e-9);
}
