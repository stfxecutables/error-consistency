# Groupings for Error Consistency

Ultimately the default error consistency metric is somewhat arbitrarily focused on pairwise
estimates of consistency. It would be more accurate to thus call it *pairwise* error consistency,
and include options for calculating e.g. *total* error consistency and $p$-error consistency for
each partition-size $k$.

That is, given $n = n_{\text{rep}} \cdot k$ predictions $\{\hat{\mathbf{y}}_i\}_{i=1}^{n}$, and true
target values $\mathbf{y}$, we get the boolean error vectors
$\mathbf{e}_i = (\hat{\mathbf{y}}_i\ == \mathbf{y})$. Get all *combinations* of size 2 of these
error vectors, i.e. the set
$\mathbf{G}^2 = \left\{G_{ij} = \{\mathbf{e}_i, \mathbf{e}_i\} \;|\; i < j\right\}$, where
$|\mathbf{G}^2| = {n\choose 2} = n_2$. Indexing these combinations with $k$ in the natural manner,
(so e.g. $k = 1$ corresponds to $(i, j) = (1, 1)$, $k = 2$ corresponds to $(i, j) = (1, 2)$ and so
on, with always $i < j$, we can write $\mathbf{G}^2 = \left\{G_k \;|\; 1 \le k \le n_2\right\}$
We then define the (pairwise) error consistency as the set of $n_2$ real values:

$$
C^2_k =
\frac{|\mathbf{e}_i \cap \mathbf{e}_j|}{|\mathbf{e}_i \cup \mathbf{e}_j|} =
\frac{\left|\bigcap_{k=1}^{2} G_k \right|}{ \left|\bigcup_{i=1}^{2} G_k \right| }
$$

Note the choice of 2 here is largely completely arbitrary, and we could just as well define
$n\choose3$ consistencies for the set
$\mathbf{G}_3 = \left\{ G_{i_1,i_2,i_3} = \{\mathbf{e}_{i_1}, \mathbf{e}_{i_2}, \mathbf{e}_{i_3}\} \;|\; i_1 < i_2 < i_j \right\}$,
with $k$ as before:

$$
C^3_k =
\frac
{|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \cap \mathbf{e}_{i_3}|}
{|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \cup \mathbf{e}_{i_3}|}
=
\frac{\left|\bigcap_{k=1}^{3} G^3_k \right|}{ \left|\bigcup_{i=1}^{3} G^3_k \right| }
$$

Likewise, we can define the $p$-wise error consistency for $\mathbf{G}_p$:

$$
C^p_k =
\frac
{|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \dots \cap \mathbf{e}_{i_p}|}
{|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \dots \cup \mathbf{e}_{i_p}|}
=
\frac{\left|\bigcap_{k=1}^{p} G^p_k \right|}{ \left|\bigcup_{i=1}^{p} G^p_k \right| },
\quad p \in \{2, 3, \dots, n\}
$$

For convenience, note we can also define $C_k^1 = 1$, which is consistent with the above definition.

Generally, we will be most interested in summary statistics of the set $\mathbf{C}^p = \{C^p_k\}$, such
as $\bar{\mathbf{C}^p}$ and $\text{var}({\mathbf{C}^p})$.

**Note:** With particularly poor and/or inconsistent classifiers, it can quite easily happen for
some values of $k$ that $\bigcup_{i=1}^{p} G^p_k = \varnothing$, which would leave the above equations
undefined. In practice, we just drop such combinations and consider only non-empty unions.

## Why $p$=2

While the "true" error consistency we would likely wish to describe as something like the average
over all valid $p$, this is obviously computationally intractable even for very small $n$ and $p$.
However, quite clearly:

$$
1 \ge C_k^1 \ge \max(\mathbf{C}^2) \ge \dots \ge \max(\mathbf{C^{p-1}}) \ge \max(\mathbf{C}^p)
$$

with equality holding very rarely (when most predictions are identical). In fact, we can make a much
stronger statement, namely:

$$ C_k^{p+1} \le \max(\mathbf{C}^p) \;\;\forall p $$

since the inclusion of more sets in the intersection can only decrease the numerator, and increase
the denominator. Thus, $\max(\mathbf{C}^2)$ provides an upper bound on the consistency. In fact,
since $\mathbf{G}^p \subset \mathbf{G}^{p+1}$ for all $p$, we necessarily for all $k$, then we can
state something even stronger:

$$
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
$$

Equality will hold only for combinations of error sets where the error sets are identical.
However, if you do the counting, sFor each $k$ and each $p$, there are in fact
${p\choose 2} = p(p-1)/2$ such unique consistencies $C^{p}_{ij}$ which are larger than $C^{p+1}_k$.
E.g. consider $p=3, 4, 5$, and that $i, j$ and $k$, $\dots$ stand in for any sequence of ascending numbers,
e.g. $(i,j,k,l,m) = (1,2,5,7,9)$. Then:

$$
C^3_{k'} = C^3_{ijk} \le (C^2_{ij},C^2_{ik}),\;(C^2_{jk})\\
C^4_{k'} = C^4_{ijkl} \le (C^2_{ij},C^2_{ik}, C^2_{il}),\; (C^2_{jk},C^2_{jl}),\; (C^2_{kl})\\
C^5_{k'} = C^5_{ijklm} \le (C^2_{ij}, C^2_{ik}, C^2_{il}), C^2_{im}),\; (C^2_{jk},C^2_{jl},
C^2_{jm}),\; (C^2_{kl}, C^2_{km}), (C^2_{lm})
$$

Clearly three are $1 + 2 + \dots (p-1) = p(p-1)/2$ values each time, since we always require our
indices to be less than each other. However, since $C^p_{k'}$ is less than *all* these values, it
is also less than the *smallest* of those values, and the *average* of those values. For different
$k$s, the smallest member of $\mathbf{C}^2$ may be the same. But there are at most $n\choose2$
unique values in $\mathbf{C}^2$.

$$
C^{p+1}_k
\le
\min(\mathbf{C}_k^2)
\le
\frac{1}{p\choose2}\sum_i^{p\choose2}C^{p}_{k_i}
 = \text{mean}(\mathbf{C}_k^2) \text{ for some }
\mathbf{C}_k^2 \subset \mathbf{C}^2
$$


Thus

$$
\sum_k^{n\choose{p+1}}C^{p+1}_k
\le
\sum_k^{n\choose{p+1}}\min(\mathbf{C}_k^2)
\le
\sum_k^{n\choose{p+1}}C^{p}_{j_k} \text{for some } j_k
$$

In fact it should be clear

## Total Error Consistency

