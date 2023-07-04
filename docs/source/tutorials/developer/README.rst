How to Contribute a New Scheduler
=================================

This tutorial guides developers and researchers to contribute a new scheduler
to Syne Tune, or to modify and extend an existing one.

We hope this information inspires you to give it a try. Please do consider
`contributing your efforts to Syne Tune <https://github.com/awslabs/syne-tune/blob/main/CONTRIBUTING.md>`__:

* Reproducible research: Syne Tune contains careful implementations of many
  baselines and SotA algorithms. Once your new method is in there, you can
  compare apples against apples (same backend, same benchmarks, same stopping
  rules) instead of apples against oranges.
* Faster and cheaper: You have a great idea for a new scheduler? Test it right
  away on a large range of benchmarks. Use Syne Tune’s
  `blackbox repository and simulator backend <../benchmarking/bm_simulator.html>`__
  in order to dramatically cut compute costs and waiting time.
* Impact: If you compared your method to a range of others, you know how hard
  it is to get full-fledged HPO code of others running. Why would it be any
  different for yours? We did a lot of the hard work already, why not benefit
  from that?
* Your code is more awesome than ours? Great! Why not contribute your backend
  or your benchmarks to Syne Tune as well?

.. note::
   In order to develop new methodology in Syne Tune, make sure to use an
   `installation from source <../../faq.html#what-are-the-different-installation-options-supported>`__.
   In particular, you need to have installed the ``dev`` dependencies.

.. toctree::
   :name: How to Contribute a New Scheduler Sections
   :maxdepth: 1

   first_example
   random_search
   trial_scheduler_api
   wrap_code
   extend_async_hb
   extend_sync_hb
   new_searcher
   documentation
