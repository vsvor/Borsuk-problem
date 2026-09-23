# A hand-checkable lower bound: sqrt(7/8)

Let P=P_3 as defined in README.md and put

\[
u=\frac{\sqrt2}{4},\qquad a=2u-\frac12,\qquad
r=\sqrt{\frac78},\qquad \mathbf1=(1,1,1).
\]
Let e_1,e_2,e_3 be the coordinate unit vectors. Consider the following 16 points:

\[
\begin{aligned}
t_0&=-u\mathbf1,\\
t_i&=u\mathbf1-2u e_i &&(1\le i\le3),\\
s_i&=-t_i &&(1\le i\le3),\\
n_i&=-2u e_i &&(1\le i\le3),\\
q_{ij}&=\tfrac12 e_i+a e_j &&(1\le i,j\le3,\ i\ne j).
\end{aligned}
\]

All belong to P. The t and s points are cube-type vertices of P; the n points
are its negative axial vertices; each q is a midpoint of an edge of a positive
square truncation face.

Assume that P is covered by four sets of diameter strictly less than r, and
color each of the 16 points by a covering set containing it. Points at distance
at least r must have different colors.

The points t_0,t_1,t_2,t_3 have all mutual distances equal to 1. Relabel their
colors as 0,1,2,3 respectively. The following exact squared distances determine
the argument (all subscripts in distinct-index formulas are distinct):

\[
\begin{aligned}
|n_i-t_j|^2&=11/8 &&(j\ne i,\ j\ne0),\\
|n_i-n_j|^2&=|s_i-s_j|^2=1,\\
|s_i-t_i|^2&=3/2,\\
|q_{ij}-t_i|^2&=|q_{ij}-s_j|^2=7/8,\\
|q_{ij}-t_0|^2&=15/8-1/\sqrt2>7/8,\\
|q_{ij}-n_i|^2&=3/2,\\
|q_{ij}-n_j|^2&=5/2-\sqrt2>7/8.
\end{aligned}
\]

Consequently n_i can have only color 0 or i, and at most one of the n_i can have
color 0. Also q_ij cannot have color 0 or i or the colors of n_i,n_j; and s_j
cannot have color j or the color of any q_ij.

## Case 1: none of the n_i has color 0

Then n_i has color i. For i!=j, q_ij must have the remaining nonzero color k.
For each j, the two incoming points q_ij (i!=j) have the two nonzero colors other
than j. Together with t_j, these rule out colors 1,2,3 for s_j. Thus all three
s_j have color 0. This contradicts |s_1-s_2|=1>r.

## Case 2: one of the n_i has color 0

By a coordinate permutation assume n_3 has color 0. Then n_1,n_2 have colors
1,2 respectively. The exclusion rules force

\[
c(q_{21})=3,\quad c(q_{31})=2,\quad
c(q_{12})=3,\quad c(q_{32})=1.
\]

Point s_1 cannot have color 1 because of t_1, color 3 because of q_21, or color 2
because of q_31. Thus it has color 0. Similarly s_2 has color 0. Again
|s_1-s_2|=1>r is a contradiction.

Both cases are impossible. Therefore every four-set cover has a part of
diameter at least r, proving

\[
\boxed{\beta_4(P_3)\ge\sqrt{7/8}=0.935414346693485\ldots.}
\]

The test suite additionally reconstructs this 16-point graph in exact
Q(sqrt(2)) arithmetic and checks its non-four-colorability, but that computation
is not needed for the two-case proof above.
