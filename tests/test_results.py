import os

from common import cleanup_after_running, path_to_test_data, simulated1
import numpy as np

import kima

def test_trend_degree_issues(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    res.plot_random_samples()
    res.hist_trend(show_prior=True, show_title=False)


def test_attributes(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    res.model
    res.priors
    res.ESS
    res.evidence
    res.information
    res.data
    res.posteriors
    res.parameter_priors


def test_methods(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)

    res.log_prior(res.posterior_sample[0])
    res.log_likelihood(res.posterior_sample[0])
    res.log_posterior(res.posterior_sample[0])
    res.maximum_likelihood_sample()
    res.map_sample()
    res.median_sample()
    res.eval_model(res.posterior_sample[0])
    res.planet_model(res.posterior_sample[0])
    res.full_model(res.posterior_sample[0])
    res.stochastic_model(res.posterior_sample[0])
    res.burst_model(res.posterior_sample[0])


def test_pickling(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    pkl = res.save_pickle()
    os.remove(pkl)


def test_pickling_compressed(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    # no compression
    pkl = res.save_pickle()
    os.remove(pkl)
    # default compression
    pkl = res.save_pickle(compress=True)
    os.remove(pkl)
    for comp in ("bz2", "gzip", "lz4", "lzma", "pickle", "zipfile"):
        pkl = res.save_pickle(compress=comp)
        os.remove(pkl)


def test_log_posterior(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)

    logp = res.log_prior(res.posterior_sample[0])
    assert np.isfinite(logp)
    logl = res.log_likelihood(res.posterior_sample[0])
    assert np.isfinite(logl)
    np.testing.assert_allclose(logl, res.posterior_lnlike[0, 1])


def test_moved_datafile(path_to_test_data):
    import shutil, os, tempfile
    from kima.pykima.utils import chdir
    from kima.pykima.cli import cli_clean
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(path_to_test_data('simulated1.txt'), tmpdir)
        with chdir(tmpdir):
            # build the model
            m = kima.RVmodel(True, 0, kima.RVData('simulated1.txt'))
            # run it
            kima.run(m, steps=100)
            # remove the datafile
            os.remove('simulated1.txt')
            assert 'simulated1.txt' not in os.listdir('.')
            # load the results
            res = kima.load_results(m)
            assert res.data.N == 40
            res.plot_random_samples()
            # cleanup
            cli_clean(check=False, output=True)


def test_np_bayes_factor_threshold():
    from dataclasses import dataclass
    from kima.pykima.analysis import np_bayes_factor_threshold

    @dataclass
    class FakeKimaResults:
        max_components: int
        Np: np.ndarray

    # no planets, 10 samples
    res = FakeKimaResults(0, np.zeros(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 0)

    # up to 1 planet, 10 samples with 0 planets
    res = FakeKimaResults(1, np.zeros(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 0)

    # up to 1 planet, 10 samples with 1 planet
    res = FakeKimaResults(1, np.ones(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 1)

    # up to 2 planets, 10 samples with 1 planet
    res = FakeKimaResults(2, np.ones(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 1)

    # probabilities 0, 1 = 0.9, 0.1
    res = FakeKimaResults(1, np.random.choice([0, 1], size=100, p=[0.9, 0.1]))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 0)


def test_out_of_order(cleanup_after_running):
    times = [
        np.array([66, 67, 104, 142, 156, 170]),
        np.array([460, 498, 763, 778, 825, 840, 1579, 1835]),
        np.array([6600, 6730]), 
        np.array([3000, 4099, 4430, 5265, 6215, 6367])
    ]
    vrad = [np.random.normal(0, 1, size=t.shape) for t in times]
    err = [0.1 * np.ones_like(t) for t in times]
    instruments = ['A', 'B', 'D', 'C']
    data = kima.RVData(times, vrad, err, instruments=instruments)
    model = kima.RVmodel(fix=True, npmax=1, data=data)
    kima.run(model, steps=500)
    res = kima.load_results(model)
    assert res.instruments == ['A', 'B', 'D', 'C']
    np.testing.assert_allclose(
        res.data.obs, 
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 3, 3]
    )
    fig = res.plot_random_samples()


def _label_switched_results(companions, n=3000, seed=42):
    """
    Fake results with `n` samples of the given companions (dicts with the median
    period P and semi-amplitude K, and the scatter of their logarithms), which
    are stored in a random order in each sample. All the other per-companion
    arrays are simple functions of the period, to detect any of them that does
    not follow the relabelling.
    """
    from types import SimpleNamespace
    from kima.pykima.results import posterior_holder

    rng = np.random.default_rng(seed)
    m = len(companions)
    P = np.column_stack([rng.lognormal(np.log(c['P']), c['sigma_lnP'], n) for c in companions])
    K = np.column_stack([rng.lognormal(np.log(c['K']), c['sigma_lnK'], n) for c in companions])
    e = rng.uniform(0, 0.5, (n, m))
    w = rng.uniform(0, 2 * np.pi, (n, m))
    phi = rng.uniform(0, 2 * np.pi, (n, m))
    truth = np.tile(np.arange(m), (n, 1))

    shuffle = np.argsort(rng.random((n, m)), axis=1)
    P, K, e, w, phi, truth = (np.take_along_axis(x, shuffle, axis=1) for x in (P, K, e, w, phi, truth))

    post = posterior_holder()
    post.P, post.K, post.e, post.w, post.φ = P, K, e, w, phi
    # as in KimaResults, some of the arrays are aliases of others
    post.ω = post.w
    post.w_deg = post.ω_deg = np.rad2deg(w)
    post.φ_deg = np.rad2deg(phi)
    post.Tp = P * phi / (2 * np.pi)
    post.i, post.i_deg = 0.1 * P, 0.2 * P
    post.W = post.Ω = 0.3 * P
    post.W_deg = post.Ω_deg = 0.4 * P

    maximum_likelihood = np.concatenate([P[0], K[0], e[0], w[0], phi[0]])
    res = SimpleNamespace(
        posteriors=post, npmax=m, max_components=m, Np=np.full(n, m),
        indices={f'planets.{name}': slice(k * m, (k + 1) * m)
                 for k, name in enumerate(('P', 'K', 'e', 'w', 'φ'))},
        maximum_likelihood_sample=lambda printit=False: maximum_likelihood.copy(),
        print_sample=lambda p: None,
    )
    return res, truth


def _relabelling_accuracy(raw, new, truth):
    """Fraction of the samples in which every companion has the label of its cluster"""
    from itertools import permutations
    permutation = (new.P[:, :, None] == raw.P[:, None, :]).argmax(-1)
    new_truth = np.take_along_axis(truth, permutation, axis=1)
    m = truth.shape[1]
    return max(np.mean(np.all(new_truth == r, axis=1)) for r in permutations(range(m)))


def test_reorder_P5_companions_with_overlapping_periods():
    import warnings
    from kima.pykima.analysis import reorder_P5

    # these can only be told apart by their semi-amplitudes
    res, truth = _label_switched_results([
        dict(P=500, sigma_lnP=0.2, K=10, sigma_lnK=0.1),
        dict(P=500, sigma_lnP=0.2, K=40, sigma_lnK=0.1),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        new = reorder_P5(res, until_detected=False)

    assert _relabelling_accuracy(res.posteriors, new, truth) > 0.99
    # by default, the labels follow the maximum likelihood sample sorted by decreasing K
    assert np.mean(new.K[:, 0] > new.K[:, 1]) > 0.99


def test_reorder_P5_sort_by_period_and_consistent_arrays():
    import warnings
    from kima.pykima.analysis import reorder_P5

    res, truth = _label_switched_results([
        dict(P=108.5, sigma_lnP=0.001, K=9.7, sigma_lnK=0.08),
        dict(P=351.6, sigma_lnP=0.002, K=32.9, sigma_lnK=0.04),
        dict(P=10233, sigma_lnP=0.04, K=93, sigma_lnK=0.04),
    ])
    raw = res.posteriors
    raw_P = raw.P.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        new = reorder_P5(res, until_detected=False, sort_by_P_after_reorder=True)

    np.testing.assert_array_equal(res.posteriors.P, raw_P)  # replace=False
    assert _relabelling_accuracy(raw, new, truth) == 1
    assert np.all(np.diff(new.P, axis=1) > 0)

    # every per-companion array follows the permutation of the periods (and only once)
    for name, factor in dict(i=0.1, i_deg=0.2, W=0.3, Ω=0.3, W_deg=0.4, Ω_deg=0.4).items():
        np.testing.assert_array_equal(getattr(new, name), factor * new.P)
    np.testing.assert_array_equal(new.w_deg, np.rad2deg(new.w))
    np.testing.assert_array_equal(new.φ_deg, np.rad2deg(new.φ))
    np.testing.assert_array_equal(new.Tp, new.P * new.φ / (2 * np.pi))
    assert new.W is new.Ω and new.ω is new.w

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reorder_P5(res, replace=True, until_detected=False, sort_by_P_after_reorder=True)
    np.testing.assert_array_equal(res.posteriors.P, new.P)