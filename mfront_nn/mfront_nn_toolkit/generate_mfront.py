#!/usr/bin/env python3
"""
generate_mfront.py
==================
Generates a complete MFront behaviour file from a JSON configuration.

Usage:
    python generate_mfront.py config.json [-o output.mfront]
    python generate_mfront.py config.json --dry-run   # print to stdout

The JSON schema is documented in configs/*.json and in docs/config_schema.md.
"""

import argparse
import json
import sys
from pathlib import Path
from textwrap import indent, dedent


# ============================================================
# Helpers
# ============================================================

def jinja_like(template: str, ctx: dict) -> str:
    """Very simple {{key}} substitution."""
    for k, v in ctx.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def indent4(s: str) -> str:
    return indent(s, "    ")


# ============================================================
# Partial generators
# ============================================================

def gen_header(cfg: dict) -> str:
    dsl = cfg["dsl"]
    lines = [
        f"@DSL {dsl};",
        f"@Behaviour {cfg['behaviour']};",
        f"@Author {cfg['author']};",
        f"@Date {cfg['date']};",
        f"@Description{{",
        f'    "{cfg["description"]}"}}',
        "",
    ]
    if cfg.get("profiling"):
        lines.append("@Profiling true;")
    if cfg.get("modelling_hypothesis"):
        lines.append(f"@ModellingHypothesis {cfg['modelling_hypothesis']};")
    if "theta" in cfg:
        lines.append(f"@Theta {cfg['theta']};")
    if "epsilon" in cfg:
        lines.append(f"@Epsilon {cfg['epsilon']};")
    if cfg.get("algorithm"):
        lines.append(f"@Algorithm {cfg['algorithm']};")
    if cfg.get("provides_tangent_operator"):
        lines.append("@ProvidesTangentOperator;")
    return "\n".join(lines)


def gen_link(cfg: dict) -> str:
    tf = cfg["tensorflow"]
    lib_path = tf["lib_path"]
    extra = tf.get("extra_link", "")
    link = f"-L{lib_path} -ltensorflow"
    if extra:
        link += f" {extra}"
    return f'@Link{{"{link}"}};'


def gen_state_variables(cfg: dict) -> str:
    lines = []
    for sv in cfg.get("state_variables", []):
        t = sv["type"]
        name = sv["name"]
        n = sv.get("array_size")
        decl = f"@StateVariable {t} {name}" + (f"[{n}]" if n else "") + ";"
        lines.append(decl)
        if sv.get("glossary_name"):
            lines.append(f'{name}.setGlossaryName("{sv["glossary_name"]}");')
        if sv.get("entry_name"):
            lines.append(f'{name}.setEntryName("{sv["entry_name"]}");')
    for av in cfg.get("aux_state_variables", []):
        lines.append(f"@AuxiliaryStateVariable {av['type']} {av['name']};")
    return "\n".join(lines)


def gen_standard_elasticity(cfg: dict) -> str:
    se = cfg.get("standard_elasticity", {})
    if not se.get("enabled"):
        return ""
    E = se["young_modulus"]
    nu = se["poisson_ratio"]
    return (
        f"@Brick StandardElasticity{{\n"
        f"    young_modulus : {E},\n"
        f"    poisson_ratio : {nu}\n"
        f"}};"
    )


# ============================================================
# @Includes block generator
# ============================================================

def _input_conversion(inp: dict, varname_suffix: str = "") -> str:
    """Generate the C++ lines that convert a named input to a cppflow::tensor."""
    name = inp["name"] + varname_suffix
    tf_input = inp["tf_input"]
    t = inp["type"]
    if t == "stensor":
        return f'cppflow::tensor {name}_tf(std::vector<double>({name}.begin(), {name}.end()), {{6}});'
    elif t == "scalar":
        return (
            f"static std::vector<double> {name}_buf(1);\n"
            f"{name}_buf[0] = {name};\n"
            f"cppflow::tensor {name}_tf({name}_buf, {{1}});"
        )
    elif t == "stensor_array":
        count = inp["count"]
        lines = [
            f"std::vector<double> {name}_data;",
            f"{name}_data.reserve({count} * 6);",
            f"for (size_t _i = 0; _i < {count}; ++_i)",
            f"    for (size_t _j = 0; _j < 6; ++_j)",
            f"        {name}_data.push_back({name}[_i][_j]);",
            f"cppflow::tensor {name}_tf({name}_data, {{{count}, 6}});",
        ]
        return "\n".join(lines)
    else:
        raise ValueError(f"Unknown input type: {t}")


def _output_unpack(out: dict, outputs_var: str = "outputs") -> str:
    """Generate C++ lines that unpack a cppflow output tensor into a TFEL type."""
    name = out["name"]
    tf_type = out["type"]
    idx = out["index"]
    ref = f"{outputs_var}[{idx}]"
    if tf_type == "stensor":
        return (
            f"nmf::Stensor {name};\n"
            f"nmf::tf_to_stensor({ref}, {name});"
        )
    elif tf_type == "st2tost2":
        return (
            f"nmf::Stensor4 {name};\n"
            f"nmf::tf_to_st2tost2({ref}, {name});"
        )
    elif tf_type == "scalar":
        return f"double {name} = {ref}.get_data<double>()[0];"
    elif tf_type == "stensor_array":
        count = out["count"]
        return (
            f"tfel::math::fsarray<{count}, nmf::Stensor> {name};\n"
            f"nmf::tf_to_stensor_array<{count}>({ref}, {name});"
        )
    elif tf_type == "st2tost2_array":
        count = out["count"]
        return (
            f"tfel::math::fsarray<{count}, nmf::Stensor4> {name};\n"
            f"nmf::tf_to_stensor4_array<{count}>({ref}, {name});"
        )
    elif tf_type == "st2tost2_matrix":
        rows = out["rows"]
        cols = out["cols"]
        return (
            f"tfel::math::fsarray<{rows * cols}, nmf::Stensor4> {name};\n"
            f"nmf::tf_to_stensor4_matrix<{rows},{cols}>({ref}, {name});"
        )
    else:
        raise ValueError(f"Unknown output type: {tf_type}")


def gen_nn_function(fn: dict) -> str:
    """Generate a single inline method for a NN function inside the struct."""
    fname = fn["name"]
    inputs = fn["inputs"]
    outputs = fn["outputs"]

    # Build return type
    ret_types = []
    for out in outputs:
        t = out["type"]
        if t == "stensor":
            ret_types.append("nmf::Stensor")
        elif t == "st2tost2":
            ret_types.append("nmf::Stensor4")
        elif t == "scalar":
            ret_types.append("double")
        elif t == "stensor_array":
            ret_types.append(f"tfel::math::fsarray<{out['count']}, nmf::Stensor>")
        elif t == "st2tost2_array":
            ret_types.append(f"tfel::math::fsarray<{out['count']}, nmf::Stensor4>")
        elif t == "st2tost2_matrix":
            ret_types.append(f"tfel::math::fsarray<{out['rows'] * out['cols']}, nmf::Stensor4>")

    ret_str = "std::tuple<\n    " + ",\n    ".join(ret_types) + ">"

    # Build parameter list
    params = []
    for inp in inputs:
        t = inp["type"]
        n = inp["name"]
        if t == "stensor":
            params.append(f"const nmf::Stensor& {n}")
        elif t == "scalar":
            params.append(f"const double& {n}")
        elif t == "stensor_array":
            cnt = inp["count"]
            params.append(f"const tfel::math::fsarray<{cnt}, nmf::Stensor>& {n}")
        else:
            params.append(f"/* TODO: {t} */ auto& {n}")

    params_str = ",\n        ".join(params)

    # Body: convert inputs
    body_lines = []
    for inp in inputs:
        body_lines.append(_input_conversion(inp))
        body_lines.append("")

    # Build model call
    model_inputs = ", ".join(
        f'{{"{inp["tf_input"]}", {inp["name"]}_tf}}' for inp in inputs
    )
    model_outputs = ", ".join(f'"{out["tf_output"]}"' for out in outputs)
    body_lines.append(f"auto outputs = model({{{{{model_inputs}}}}},")
    body_lines.append(f"                     {{{{{model_outputs}}}}});")
    body_lines.append("")

    # Unpack outputs
    for out in outputs:
        body_lines.append(_output_unpack(out, "outputs"))
        body_lines.append("")

    # Return statement
    ret_names = ", ".join(out["name"] for out in outputs)
    body_lines.append(f"return {{{ret_names}}};")

    body = "\n".join(body_lines)

    method = (
        f"inline {ret_str}\n"
        f"{fname}(\n"
        f"        {params_str})\n"
        f"{{\n"
        f"{indent4(body)}\n"
        f"}}"
    )
    return method


def gen_includes(cfg: dict) -> str:
    tf = cfg["tensorflow"]
    model_path = tf["model_path"]
    fns = cfg["nn_functions"]

    methods = "\n\n".join(gen_nn_function(fn) for fn in fns)

    struct_body = (
        f"inline NN(const std::string& model_path)\n"
        f"    : model(model_path) {{}}\n\n"
        + methods
        + "\n\nprivate:\n    cppflow::model model;"
    )

    struct_code = f"struct NN {{\n{indent4(struct_body)}\n}};"

    singleton = (
        f"inline NN& getNN() {{\n"
        f'    static NN nn("{model_path}");\n'
        f"    return nn;\n"
        f"}}"
    )

    # Check if an external header is used instead
    if tf.get("use_external_header"):
        ext = tf["use_external_header"]
        return (
            f"@Includes\n{{\n"
            f'#include "{ext}"\n'
            f"}}"
        )

    inner = (
        "#ifndef LIB_MFRONT_TENSORFLOW\n"
        "#define LIB_MFRONT_TENSORFLOW 1\n\n"
        '#include "cppflow/cppflow.h"\n'
        '#include "nn_mfront.hxx"\n'
        "#include <vector>\n\n"
        + struct_code
        + "\n\n"
        + singleton
        + "\n\n"
        "#endif /* LIB_MFRONT_TENSORFLOW */"
    )

    return f"@Includes\n{{\n{indent4(inner)}\n}}"


# ============================================================
# @Integrator / @ComputeFinalStress / @TangentOperator templates
# ============================================================

def gen_integrator_nl_elasticity(cfg: dict) -> str:
    return dedent("""\
        @Integrator {
            static NN nn(MODEL_PATH);
            sig = nn.stress(eto + deto);
            Dt  = nn.tangentop(eto + deto);
        }""").replace("MODEL_PATH", f'"{cfg["tensorflow"]["model_path"]}"')


def gen_integrator_standard_elasticity_phi2_sigma(cfg: dict) -> str:
    return dedent("""\
        @Integrator
        {
            auto& nn = getNN();
            feel += devp;
            dfeel_ddevp = Stensor4::Id();
            auto [depsp, ddepsp_dsig] = nn.phi2(sig);
            fevp -= dt * depsp;
            const auto De = lambda * Stensor4::IxI() + 2 * mu * Stensor4::Id();
            dfevp_ddeel -= dt * theta * ddepsp_dsig * De;
        }""")


def gen_integrator_standard_elasticity_phi2_sigma_p(cfg: dict) -> str:
    return dedent("""\
        @Integrator
        {
            auto& nn = getNN();
            feel += devp;
            dfeel_ddevp = Stensor4::Id();
            auto [depsp, dp_rate, ddepsp_dsig, ddepsp_dp, ddp_dsig, ddp_dp]
                = nn.phi2(sig, p + theta * dp);
            fevp -= dt * depsp;
            fp   -= dt * dp_rate;
            const auto De = lambda * Stensor4::IxI() + 2 * mu * Stensor4::Id();
            dfevp_ddeel -= dt * theta * ddepsp_dsig * De;
            dfevp_ddp   -= dt * theta * ddepsp_dp;
            dfp_ddeel   -= dt * theta * (ddp_dsig | De);
            dfp_ddp     -= dt * theta * ddp_dp;
            ++niter;
        }""")


def gen_integrator_standard_elasticity_phi2_sigma_p_alpha(cfg: dict) -> str:
    return dedent("""\
        @Integrator
        {
            auto& nn = getNN();
            feel += devp;
            dfeel_ddevp = Stensor4::Id();
            auto [depsp, dp_rate, dalpha_rate,
                  ddepsp_dsig, ddepsp_dp, ddepsp_dalpha,
                  ddp_dsig,    ddp_dp,    ddp_dalpha,
                  ddalpha_dsig, ddalpha_dp, ddalpha_dalpha]
                = nn.phi2(sig, p + theta * dp, alpha + theta * dalpha);
            fevp   -= dt * depsp;
            fp     -= dt * dp_rate;
            falpha -= dt * dalpha_rate;
            const auto De = lambda * Stensor4::IxI() + 2 * mu * Stensor4::Id();
            dfevp_ddeel   -= dt * theta * ddepsp_dsig   * De;
            dfevp_ddp     -= dt * theta * ddepsp_dp;
            dfevp_ddalpha -= dt * theta * ddepsp_dalpha;
            dfp_ddeel     -= dt * theta * (ddp_dsig   | De);
            dfp_ddp       -= dt * theta * ddp_dp;
            dfp_ddalpha   -= dt * theta * (ddp_dalpha | De);
            dfalpha_ddeel   -= dt * theta * ddalpha_dsig   * De;
            dfalpha_ddp     -= dt * theta * ddalpha_dp;
            dfalpha_ddalpha -= dt * theta * ddalpha_dalpha;
            ++niter;
        }""")


def gen_integrator_ngsm_single_alpha(cfg: dict) -> str:
    return dedent("""\
        @Integrator
        {
            auto& nn = getNN();
            auto [phi2_val, dphi2_dx, dphi2_dy]
                = nn.phi2(eto + theta * deto, alpha + theta * dalpha);
            falpha         -= dt * phi2_val;
            dfalpha_ddalpha -= theta * dt * dphi2_dy;
        }

        @ComputeFinalStress
        {
            auto& nn = getNN();
            auto [sig_val, dphi1_dx, dphi1_dy] = nn.phi1(eto + deto, alpha);
            sig = sig_val;
        }

        @TangentOperator
        {
            auto& nn = getNN();
            auto [sig_val, dphi1_dx, dphi1_dy] = nn.phi1(eto + deto, alpha);
            auto [phi2_val, dphi2_dx, dphi2_dy]
                = nn.phi2(eto + theta * deto, alpha - (1.0 - theta) * dalpha);
            dfalpha_ddeto = -theta * dt * dphi2_dx;
            auto ddalpha_ddeto = Stensor4{};
            getIntegrationVariablesDerivatives_eto(ddalpha_ddeto);
            Dt = 2 * dphi1_dx + dphi1_dy * ddalpha_ddeto;
        }""")


def gen_integrator_ngsm_branch_alpha(cfg: dict) -> str:
    n = cfg.get("branch_count", 3)
    return dedent(f"""\
        @Integrator
        {{
            auto alpha_theta = alpha;
            for (size_t i = 0; i < {n}; ++i)
                alpha_theta[i] += theta * dalpha[i];
            auto& nn = getNN();
            auto [phi2_val, dphi2_deto, dphi2_dalpha]
                = nn.phi2(eto + theta * deto, alpha_theta);
            for (size_t i = 0; i < {n}; ++i) {{
                falpha(i) -= dt * phi2_val[i];
                for (size_t j = 0; j < {n}; ++j)
                    dfalpha_ddalpha(i, j) -= theta * dt * dphi2_dalpha[i * {n} + j];
            }}
        }}

        @ComputeFinalStress
        {{
            auto& nn = getNN();
            auto [stress, dphi1_deto, dphi1_dalpha] = nn.phi1(eto + deto, alpha);
            sig = stress;
        }}

        @TangentOperator
        {{
            auto& nn = getNN();
            auto [stress, dphi1_deto, dphi1_dalpha] = nn.phi1(eto + deto, alpha);
            auto alpha_theta = alpha;
            for (size_t i = 0; i < {n}; ++i)
                alpha_theta[i] -= (1.0 - theta) * dalpha[i];
            auto [phi2_val, dphi2_deto, dphi2_dalpha]
                = nn.phi2(eto + theta * deto, alpha_theta);
            for (size_t i = 0; i < {n}; ++i)
                dfalpha_ddeto(i) = -theta * dt * dphi2_deto(i);
            auto ddalpha_ddeto = tvector<{n}u, Stensor4>{{}};
            getIntegrationVariablesDerivatives_eto(ddalpha_ddeto);
            Dt = Stensor4{{}};
            Dt = 2 * dphi1_deto;
            for (size_t i = 0; i < {n}; ++i)
                Dt += dphi1_dalpha(i) * ddalpha_ddeto(i);
        }}""")


INTEGRATOR_GENERATORS = {
    "nl_elasticity":                              gen_integrator_nl_elasticity,
    "standard_elasticity_phi2_sigma":             gen_integrator_standard_elasticity_phi2_sigma,
    "standard_elasticity_phi2_sigma_p":           gen_integrator_standard_elasticity_phi2_sigma_p,
    "standard_elasticity_phi2_sigma_p_alpha":     gen_integrator_standard_elasticity_phi2_sigma_p_alpha,
    "ngsm_single_alpha":                          gen_integrator_ngsm_single_alpha,
    "ngsm_branch_alpha":                          gen_integrator_ngsm_branch_alpha,
}


# ============================================================
# Top-level assembler
# ============================================================

def generate(cfg: dict) -> str:
    parts = []

    # 1 – DSL header
    parts.append(gen_header(cfg))
    parts.append("")

    # 2 – @Link
    parts.append(gen_link(cfg))
    parts.append("")

    # 3 – @Includes
    parts.append(gen_includes(cfg))
    parts.append("")

    # 4 – State variables
    sv_block = gen_state_variables(cfg)
    if sv_block:
        parts.append(sv_block)
        parts.append("")

    # 5 – StandardElasticity brick
    se_block = gen_standard_elasticity(cfg)
    if se_block:
        parts.append(se_block)
        parts.append("")

    # 6 – Integrator / ComputeFinalStress / TangentOperator
    tmpl = cfg.get("integrator_template", "")
    if tmpl in INTEGRATOR_GENERATORS:
        parts.append(INTEGRATOR_GENERATORS[tmpl](cfg))
    else:
        parts.append(
            f"// TODO: integrator_template '{tmpl}' not recognised.\n"
            "// Implement @Integrator manually or add a generator in INTEGRATOR_GENERATORS.\n"
            "@Integrator\n{\n    // ...\n}"
        )

    return "\n".join(parts) + "\n"


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a MFront behaviour file from a JSON config."
    )
    parser.add_argument("config", help="Path to the JSON configuration file.")
    parser.add_argument(
        "-o", "--output",
        help="Output .mfront file path. Defaults to <behaviour>.mfront.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the generated file to stdout without writing it.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        cfg = json.load(f)

    output = generate(cfg)

    if args.dry_run:
        print(output)
        return

    out_path = Path(args.output) if args.output else Path(cfg["behaviour"] + ".mfront")
    out_path.write_text(output)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
