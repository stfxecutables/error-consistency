.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5555408.svg
   :target: https://doi.org/10.5281/zenodo.5555408

Quickstart
============

To fit a standard scikit-learn classifier, or any classifier with either ``.fit`` or ``.train`` and
``.predict`` or ``.test`` methods, using `error_consistency.consistency.ErrorConsistencyKFoldHoldout`

.. testcode::

   import numpy as np
   from error_consistency.consistency import ErrorConsistencyKFoldHoldout as ErrorConsistency
   from sklearn.neighbors import KNeighborsClassifier as KNN

   x = np.random.uniform(0, 1, size=[100, 5])
   y = np.random.randint(0, 3, size=[100])
   x_test = np.random.uniform(0, 1, size=[20, 5])
   y_test = np.random.randint(0, 3, size=[20])

   knn_args = dict(n_neighbors=5, n_jobs=1)
   errcon = ErrorConsistency(KNN, x, y, n_splits=5, model_args=knn_args)
   results = errcon.evaluate(
      x_test,
      y_test,
      repetitions=10,
      show_progress=True,
      parallel_reps=True,
      loo_parallel=False,
      turbo=True
   )

   # 10 reps * 5 splits = 50 errors sets
   print(len(results.consistencies))  # 50*(50-1) // 2

.. testoutput::
   :hide:

   1225

To evaluate the error consistency of a set of predictions on a test set with `error_consistency.functional.error_consistencies`:

.. testcode::

   import numpy as np
   from error_consistency.functional import error_consistencies
   from sklearn.neighbors import KNeighborsClassifier as KNN

   # random training set
   N_TRAIN = 10
   x_trains = [np.random.uniform(0, 1, size=[100, 5]) for _ in range(N_TRAIN)]
   y_trains = [np.random.randint(0, 3, size=[100]) for _ in range(N_TRAIN)]
   x_test = np.random.uniform(0, 1, size=[50, 5])
   y_test = np.random.randint(0, 3, size=[50])
   y_preds = [KNN(5).fit(x, y).predict(x_test) for x, y in zip(x_trains, y_trains)]

   # only grab consistencies and matrix
   consistencies, matrix = error_consistencies(y_preds, y_test)[0:2]
   print(matrix.shape)  # (10, 10)

.. testoutput::
   :hide:

   (10, 10)
