
Error Consistency Theory
===========================

**NOTE** This section is currently a work in progress.

----------------------------------
Choice of Groupings
----------------------------------

Ultimately the default error consistency metric is somewhat arbitrarily focused on pairwise
estimates of consistency. It would be more accurate to thus call it *pairwise* error consistency,
and include options for calculating e.g. *total* error consistency and :math:`p`-error consistency for
each partition-size :math:`k`.

That is, given :math:`n = n_{\text{rep}} \cdot k` predictions :math:`\{\hat{\mathbf{y}}_i\}_{i=1}^{n}`, and true
target values :math:`\mathbf{y}`, we get the boolean error vectors
:math:`\mathbf{e}_i = (\hat{\mathbf{y}}_i\ == \mathbf{y})`. Get all *combinations* of size 2 of these
error vectors, i.e. the set
:math:`\mathbf{G}^2 = \left\{G_{ij} = \{\mathbf{e}_i, \mathbf{e}_i\} \;|\; i < j\right\}`, where
:math:`|\mathbf{G}^2| = {n\choose 2} = n_2`. Indexing these combinations with :math:`k` in the natural manner,
(so e.g. :math:`k = 1` corresponds to :math:`(i, j) = (1, 1)`, :math:`k = 2` corresponds to
:math:`(i, j) = (1, 2)` and so
on, with always :math:`i < j`, we can write :math:`\mathbf{G}^2 = \left\{G_k \;|\; 1 \le k \le n_2\right\}`.
We then define the (pairwise) error consistency as the set of :math:`n_2` real values:

.. math::

  C^2_k =
  \frac{|\mathbf{e}_i \cap \mathbf{e}_j|}{|\mathbf{e}_i \cup \mathbf{e}_j|} =
  \frac{\left|\bigcap_{k=1}^{2} G_k \right|}{ \left|\bigcup_{i=1}^{2} G_k \right| }

Note the choice of 2 here is largely completely arbitrary, and we could just as well define
:math:`n\choose3` consistencies for the set
:math:`\mathbf{G}_3 = \left\{ G_{i_1,i_2,i_3} = \{\mathbf{e}_{i_1}, \mathbf{e}_{i_2}, \mathbf{e}_{i_3}\} \;|\; i_1 < i_2 < i_j \right\}`,
with :math:`k` as before:

.. math::

   C^3_k =
   \frac
   {|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \cap \mathbf{e}_{i_3}|}
   {|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \cup \mathbf{e}_{i_3}|}
   =
   \frac{\left|\bigcap_{k=1}^{3} G^3_k \right|}{ \left|\bigcup_{i=1}^{3} G^3_k \right| }

Likewise, we can define the :math:`p`-wise error consistency for :math:`\mathbf{G}_p`:

.. math::

   C^p_k =
   \frac
   {|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \dots \cap \mathbf{e}_{i_p}|}
   {|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \dots \cup \mathbf{e}_{i_p}|}
   =
   \frac{\left|\bigcap_{k=1}^{p} G^p_k \right|}{ \left|\bigcup_{i=1}^{p} G^p_k \right| },
   \quad p \in \{2, 3, \dots, n\}

For convenience, note we can also define :math:`C_k^1 = 1`, which is consistent with the above definition.

Generally, we will be most interested in summary statistics of the set :math:`\mathbf{C}^p = \{C^p_k\}`, such
as :math:`\bar{\mathbf{C}^p}` and :math:`\text{var}({\mathbf{C}^p})`.

**Note:** With particularly poor and/or inconsistent classifiers, it can quite easily happen for
some values of :math:`k` that :math:`\bigcup_{i=1}^{p} G^p_k = \varnothing`, which would leave the above equations
undefined. In practice, we just drop such combinations and consider only non-empty unions.

----------------------------------
Why p=2
----------------------------------

While the "true" error consistency we would likely wish to describe as something like the average
over all valid :math:`p`, this is obviously computationally intractable even for very small :math:`n` and :math:`p`.
However, quite clearly:

.. math::

   1 \ge C_k^1 \ge \max(\mathbf{C}^2) \ge \dots \ge \max(\mathbf{C^{p-1}}) \ge \max(\mathbf{C}^p)

with equality holding very rarely (when most predictions are identical). In fact, we can make a much
stronger statement, namely:

.. math::

   C_k^{p+1} \le \max(\mathbf{C}^p) \;\;\forall p

since the inclusion of more sets in the intersection can only decrease the numerator, and increase
the denominator. Thus, :math:`\max(\mathbf{C}^2)` provides an upper bound on the consistency. In fact,
since :math:`\mathbf{G}^p \subset \mathbf{G}^{p+1}` for all :math:`p`, we necessarily for all :math:`k`, then we can
state something even stronger:

.. math::

   C^{p+1}_k =
   \frac
   {|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \dots \cap \mathbf{e}_{i_{p+1}}|}
   {|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \dots \cup \mathbf{e}_{i_{p+1}}|}
   \le
   \frac
   {|\mathbf{e}_{i} \cap \mathbf{e}_{j}|}
   {|\mathbf{e}_{i} \cup \mathbf{e}_{j}|}
   =
   C^{p}_{ij} \text{for all } i, j \in \{i_1, \dots, i_p+1\} \text{ where } i < j

Equality will hold only for combinations of error sets where the error sets are identical.
However, if you do the counting, sFor each :math:`k` and each :math:`p`, there are in fact
:math:`{p\choose 2} = p(p-1)/2` such unique consistencies :math:`C^{p}_{ij}` which are larger than :math:`C^{p+1}_k`.
E.g. consider :math:`p=3, 4, 5`, and that :math:`i, j` and :math:`k`, :math:`\dots` stand in for any sequence of ascending numbers,
e.g. :math:`(i,j,k,l,m) = (1,2,5,7,9)`. Then:

.. math::

   \begin{align}
   C^3_{k'} &= C^3_{ijk} \le (C^2_{ij},C^2_{ik}),\;(C^2_{jk})\\
   C^4_{k'} &= C^4_{ijkl} \le (C^2_{ij},C^2_{ik}, C^2_{il}),\; (C^2_{jk},C^2_{jl}),\; (C^2_{kl})\\
   C^5_{k'} &= C^5_{ijklm} \le (C^2_{ij}, C^2_{ik}, C^2_{il}), C^2_{im}),\; (C^2_{jk},C^2_{jl},
   C^2_{jm}),\; (C^2_{kl}, C^2_{km}), (C^2_{lm})
   \end{align}

Clearly three are :math:`1 + 2 + \dots (p-1) = p(p-1)/2` values each time, since we always require our
indices to be less than each other. However, since :math:`C^p_{k'}` is less than *all* these values, it
is also less than the *smallest* of those values, and the *average* of those values. For different
values of :math:`k`, the smallest member of :math:`\mathbf{C}^2` may be the same. But there are at
most :math:`n\choose2` unique values in :math:`\mathbf{C}^2`.

.. math::

   C^{p+1}_k
   \le
   \min(\mathbf{C}_k^2)
   \le
   \frac{1}{p\choose2}\sum_i^{p\choose2}C^{p}_{k_i}
   = \text{mean}(\mathbf{C}_k^2) \text{ for some }
   \mathbf{C}_k^2 \subset \mathbf{C}^2


Thus

.. math::

   \sum_k^{n\choose{p+1}}C^{p+1}_k
   \le
   \sum_k^{n\choose{p+1}}\min(\mathbf{C}_k^2)
   \le
   \sum_k^{n\choose{p+1}}C^{p}_{j_k} \text{for some } j_k

In fact it should be clear

## Total Error Consistency

