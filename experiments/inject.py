"""Activation-addition sweeps: inject mu, and inject the prompt's own shift."""

import sys
import argparse
import json
import os
import numpy as np
import torch
from transformers import logging
from experiments._args import add_generation, add_injection, add_limit, add_model, add_steering_tag
from src.evaluation import load_resid_norms
from src.data import MU_SOURCES, injection_arms, select_eval_items, source_for_tag
from src.model import ActivationHooks, hook_layer_for, load_mu
from src.model import load_model, num_layers
from src.paths import mu_injection_dir
from src.evaluation import InjectionScorer
from src.utils import seed_everything
from src.model import generate as gen_round0
from src.variants import eval_set_for, round0_prompt, round0_variant_for, system_prompt_for
import pickle
from src.evaluation import load_resid_norms, resolve_coef
from src.data import EVAL_SETS, select_eval_items
from src.model import flush_results, generate, load_results
from src.model import ActivationHooks, hook_layer_for, unit
from src.paths import isc_dir, isc_item, shift_injection_dir
from src.variants import (
    attribute_for, eval_data_type, eval_set_for, round0_prompt, round0_variant_for,
    system_prompt_for,
)


def _cli_mu(argv):
    logging.set_verbosity_error()

    def run_one(model, tokenizer, hooks, mu, norms, data_type, alpha, prompts, args, scorer):
        """Generate one item, with the coefficient scaled by the layer's residual norm."""
        out_dir = mu_injection_dir(args.model, args.steering_tag, data_type, alpha)
        os.makedirs(out_dir, exist_ok=True)
        n_layers = num_layers(model)
        layers = list(range(0, n_layers, args.layer_stride))
        system = system_prompt_for(args.variant)

        def gen(batch):
            return gen_round0(model, tokenizer, [round0_prompt(args.variant, p) for p in batch],
                              args.max_new_tokens, args.min_new_tokens, system_prompt=system)

        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            ids = list(range(start, start + len(batch)))

            paths = {d: os.path.join(out_dir, f"{d:05d}.json") for d in ids}
            results = {
                d: (json.load(open(p)) if os.path.exists(p) else {"prompt": prompts[d]})
                for d, p in paths.items()
            }

            def flush():
                for d in ids:
                    with open(paths[d], "w") as f:
                        json.dump(results[d], f)

            if not all(scorer.key("baseline") in results[d] for d in ids):
                hooks.clear()
                texts = gen(batch)
                first, second = scorer(texts, prompts=batch)
                for i, d in enumerate(ids):
                    results[d]["baseline_text"] = texts[i]
                    scorer.store(results[d], "baseline", i, first, second)
                flush()

            for layer in layers:
                key = f"steered_L{layer}"
                if all(scorer.key(key) in results[d] for d in ids):
                    continue

                coef = alpha * norms[layer]
                hooks.clear()
                hooks.add(hook_layer_for(layer, args.inject_offset, n_layers), mu[layer], alpha=coef)
                texts = gen(batch)
                first, second = scorer(texts, prompts=batch)
                hooks.clear()

                for i, d in enumerate(ids):
                    results[d][f"{key}_text"] = texts[i]
                    scorer.store(results[d], key, i, first, second)
                    results[d][f"{key}_coef"] = coef
                    results[d][f"{key}_resid_norm"] = norms[layer]
                flush()
                torch.cuda.empty_cache()

            base = np.mean([results[d][scorer.key("baseline")] for d in ids])
            print(f"[{data_type} a={alpha:+g}] {ids[-1] + 1}/{len(prompts)} "
                  f"baseline_{scorer.primary}={base:.3f}", flush=True)

    def main(args):
        seed_everything(87, benchmark=True)

        source = source_for_tag(args.steering_tag)
        attribute = MU_SOURCES[source]["attribute"]
        if args.variant is None:
            args.variant = round0_variant_for(attribute)
        eval_set = eval_set_for(args.variant)
        if eval_set != MU_SOURCES[source]["eval_set"]:
            raise ValueError(
                f"--variant {args.variant!r} evaluates on {eval_set!r} but steering tag "
                f"{args.steering_tag!r} is built for {MU_SOURCES[source]['eval_set']!r}"
            )

        model, tokenizer = load_model(
            args.model, dtype=torch.float16, device_map="auto"
        )
        n_layers = num_layers(model)
        mu = load_mu(args.model, args.steering_tag, model.device)
        norms = load_resid_norms(args.model, n_layers, attribute)
        hooks = ActivationHooks(model, apply_during_decode=args.apply_during_decode)

        scorer = InjectionScorer(attribute)

        print(f"source={source}  attribute={attribute}  eval_set={eval_set}  variant={args.variant}")
        print("layer  ||x||   coef at alpha=" + " ".join(f"{a:g}" for a in args.alphas))
        for l in range(0, n_layers, args.layer_stride):
            print(f"{l:5d}  {norms[l]:6.1f}  " + "  ".join(f"{a * norms[l]:7.2f}" for a in args.alphas))

        for alpha_abs in args.alphas:
            for data_type, alpha in injection_arms(source, alpha_abs):
                prompts = select_eval_items(args.data_dir, data_type, args.limit, eval_set=eval_set)
                run_one(model, tokenizer, hooks, mu, norms, data_type, alpha, prompts, args, scorer)

        hooks.remove()

    p = argparse.ArgumentParser()
    add_model(p)
    add_steering_tag(p)
    p.add_argument("--data_dir", default=None,
                   help="defaults to the eval set registered for --steering_tag's attribute")
    p.add_argument("--variant", default=None,
                   help="only its ROUND-0 templates are used -- there is no revise instruction in "
                        "this experiment. It exists so this sweep generates under a context "
                        "byte-identical to the transplant's and to ISC round 0, which is what makes "
                        "the mu-vs-ell_bar comparison in Fig 2 a comparison of VECTORS and not of "
                        "prompts. Defaults to the attribute's registered round-0 "
                        "variant (src.variants.ROUND0_VARIANT), i.e. strong_detox for toxicity "
                        "and moralize_strong for moral.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="FRACTION of median||x^(l)|| to inject; 1.0 = a perturbation the size of "
                        "the residual stream itself. The effective range "
                        "starts around 0.25 and the detox effect only saturates near 1.0-1.5. "
                        "Anything below ~0.25 is in the dead zone and returns a flat curve.")
    add_limit(p, 200)
    add_generation(p, batch_size=10, max_new_tokens=200, min_new_tokens=5)
    add_injection(p)
    main(p.parse_args(argv))


def _cli_shift(argv):
    def estimate_ell_bar(model_name, variant, n_train):
        """mean_i [ h_1(i) - h_0(i) ] over the train split."""
        train_dir = isc_dir(model_name, variant, "train")
        if not os.path.isdir(train_dir):
            sys.exit(
                f"missing {train_dir}\nRun:\n  python -m experiments.pipeline self_correction --model {model_name} "
                f"--variants {variant} --split train --num_rounds 2 --num_data {n_train}"
            )

        shifts = []
        for i in range(n_train):
            try:
                with open(isc_item(model_name, variant, 0, i, "train"), "rb") as f:
                    h0 = pickle.load(f)["avg_output_hidden"]
                with open(isc_item(model_name, variant, 1, i, "train"), "rb") as f:
                    h1 = pickle.load(f)["avg_output_hidden"]
            except FileNotFoundError:
                break
            shifts.append(h1.astype(np.float32) - h0.astype(np.float32))

        if not shifts:
            sys.exit(f"no train items found under {train_dir}")
        print(f"ell_bar estimated from {len(shifts)} train items")
        return np.mean(np.stack(shifts), axis=0)

    def run(args):
        seed_everything(87, benchmark=True)

        variant = args.variant
        data_type = eval_data_type(variant)
        eval_set = eval_set_for(variant)
        attribute = attribute_for(variant)

        pools = EVAL_SETS[eval_set]["data_types"]
        arm = None if len(pools) > 1 or variant == round0_variant_for(attribute) else variant
        if arm:
            print(f"single-pool eval set: tagging transplant dirs with arm={arm!r}")

        ell_bar = estimate_ell_bar(args.model, variant, args.n_train)

        model, tokenizer = load_model(
            args.model, dtype=torch.float16, device_map="auto"
        )
        n_layers = num_layers(model)
        hooks = ActivationHooks(model, apply_during_decode=args.apply_during_decode)
        scorer = InjectionScorer(attribute)

        ell_hat = {
            l: unit(torch.from_numpy(ell_bar[l + 1])).to(model.device) for l in range(n_layers)
        }
        ell_norms = {l: float(np.linalg.norm(ell_bar[l + 1])) for l in range(n_layers)}

        resid_norms = load_resid_norms(args.model, n_layers, attribute)

        prompts = select_eval_items(args.data_dir, data_type, args.limit, eval_set=eval_set)
        layers = list(range(0, n_layers, args.layer_stride))

        print(f"alpha_mode={args.alpha_mode}")
        print(f"{'layer':>5} {'||x||':>8} {'||ell_bar||':>12}  " +
              "  ".join(f"a={a:g}" for a in args.alphas))
        for l in layers:
            coefs = [resolve_coef(a, l, resid_norms, args.alpha_mode) for a in args.alphas]
            print(f"{l:>5} {resid_norms[l]:>8.1f} {ell_norms[l]:>12.2f}  " +
                  "  ".join(f"{c:7.2f}" for c in coefs))

        for alpha in args.alphas:
            for layer in layers:
                out_dir = shift_injection_dir(
                    args.model, args.steering_tag, data_type, alpha, layer, args.alpha_mode,
                    arm=arm,
                )
                done = os.path.join(out_dir, "_complete")
                if os.path.exists(done):
                    try:
                        prev = int(open(done).read().strip() or 0)
                    except ValueError:
                        prev = 0
                    if prev >= len(prompts):
                        continue
                    print(f"  re-running {os.path.basename(out_dir)}: marked complete at n={prev}, "
                          f"but this run wants n={len(prompts)}", flush=True)

                coef = resolve_coef(alpha, layer, resid_norms, args.alpha_mode)

                for start in range(0, len(prompts), args.batch_size):
                    batch = prompts[start : start + args.batch_size]
                    ids = list(range(start, start + len(batch)))
                    results, paths = load_results(out_dir, ids, prompts)

                    if all(scorer.key("transplant") in results[d] for d in ids):
                        continue

                    hooks.clear()
                    hooks.add(
                        hook_layer_for(layer, args.inject_offset, n_layers),
                        ell_hat[layer],
                        alpha=coef,
                    )
                    texts = generate(
                        model, tokenizer, [round0_prompt(variant, p) for p in batch],
                        args.max_new_tokens, args.min_new_tokens,
                        system_prompt=system_prompt_for(variant),
                    )
                    hooks.clear()

                    first, second = scorer(texts, prompts=batch)
                    for i, d in enumerate(ids):
                        scorer.store(results[d], "transplant", i, first, second)
                        results[d].update(
                            transplant_text=texts[i],
                            alpha=alpha,
                            alpha_mode=args.alpha_mode,
                            coef=coef,
                            layer=layer,
                            ell_bar_norm=ell_norms[layer],
                            resid_norm=resid_norms[layer],
                        )
                    flush_results(results, paths)
                    torch.cuda.empty_cache()

                mean_score = np.mean([
                    json.load(open(os.path.join(out_dir, f"{d:05d}.json")))[scorer.key("transplant")]
                    for d in range(len(prompts))
                ])
                with open(done, "w") as f:
                    f.write(str(len(prompts)))
                print(f"[alpha={alpha:g} L={layer}] coef={coef:.2f}  "
                      f"{scorer.primary}={mean_score:.3f}  "
                      f"||x||={resid_norms[layer]:.1f}  ||ell_bar||={ell_norms[layer]:.2f}",
                      flush=True)

        hooks.remove()

    p = argparse.ArgumentParser()
    add_model(p)
    p.add_argument("--data_dir", default=None,
                   help="defaults to the eval set registered for --variant")
    add_steering_tag(p)
    p.add_argument("--variant", default="strong_detox")
    p.add_argument("--alpha_mode", choices=["norm", "absolute"], default="norm",
                   help="'norm': alpha is a fraction of median||x^(l)|| at the injection site "
                        "(requires measure_residual_norms; comparable across layers and against alpha's "
                        "steering sweep). 'absolute': the published parameterisation, kept only "
                        "so the original numbers stay reproducible.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="In norm mode, matching --alphas. 1.0 displaces the residual stream by "
                        "its own magnitude. The old default (1 2 4, absolute) is the grid that "
                        "produced a null result -- do not restore it.")
    p.add_argument("--n_train", type=int, default=200, help="train items backing ell_bar")
    add_limit(p, 500)
    add_generation(p, batch_size=10, max_new_tokens=200, min_new_tokens=5)
    add_injection(p)
    run(p.parse_args(argv))


_COMMANDS = {"mu": _cli_mu, "shift": _cli_shift}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in _COMMANDS:
        sys.exit("usage: python -m experiments.inject <cmd>  (cmd: " + ", ".join(_COMMANDS) + ")")
    _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
