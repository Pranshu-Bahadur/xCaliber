# A Profiled Tensor Realization of CuTe Layouts

## The whole formulation

Fix a nonzero commutative unital ring $R$ and an $R$-module $M$. For ordinary integer-address layouts, take

$$
R=M=\mathbb Z.
$$

A hierarchical shape $S$ determines a finite set of natural coordinates

$$
\mathsf C(S),
$$

and therefore a finite free coordinate module

$$
\mathsf V_R(S):=R[\mathsf C(S)].
$$

At every tuple node $S=(S_0,\ldots,S_{r-1})$, there is a canonical basis-preserving isomorphism

$$
\boxed{
\mathsf V_R(S)
\xrightarrow[\cong]{\;\chi_S\;}
\mathsf V_R(S_0)\otimes_R\cdots\otimes_R\mathsf V_R(S_{r-1}).
}
$$

The complete family of these isomorphisms, one for every node of $S$, is the **profiled tensor realization** of $S$. The underlying module alone does not remember the nesting; the profile together with its tensor-decomposition arrows does.

A stride $D$ congruent to $S$ determines the CuTe natural-coordinate function

$$
\ell_{S,D}:\mathsf C(S)\longrightarrow U(M),
\qquad
c\longmapsto \sum_{\lambda\in\operatorname{Leaf}(S)}[c_\lambda]_R d_\lambda,
$$

where $U(M)$ is the underlying set of $M$. By the free-module universal property, it extends uniquely to an $R$-linear map

$$
\boxed{
\widetilde\ell_{S,D}:\mathsf V_R(S)\longrightarrow M,
\qquad
e_c\longmapsto \ell_{S,D}(c).
}
$$

More finely, this map factors through the leaf-component module

$$
\mathsf Q_R(S):=R[\operatorname{Leaf}(S)]:
$$

$$
\boxed{
\mathsf V_R(S)
\xrightarrow{\;\widetilde j_S\;}
\mathsf Q_R(S)
\xrightarrow{\;d_D\;}
M,
\qquad
\widetilde\ell_{S,D}=d_D\circ\widetilde j_S.
}
$$

Here

$$
\operatorname{rank}_R\mathsf Q_R(S)=\operatorname{len}(S),
\qquad
\operatorname{rank}_R\mathsf V_R(S)=|S|.
$$

The component module uses direct sums; the state module uses tensor products.

After choosing the colexicographic enumeration of coordinates, the ordinary one-dimensional layout function is the composite

$$
\boxed{
\begin{array}{ccccc}
[\,|S|\,]
&\xrightarrow[\cong]{\operatorname{idx2crd}_S}&
\mathsf C(S)
&\xrightarrow{\eta_S}&
U\mathsf V_R(S)
\xrightarrow{\,U\widetilde\ell_{S,D}\,}
U(M),
\\[2mm]
i&\longmapsto&c&\longmapsto&e_c\longmapsto\ell_{S,D}(c).
\end{array}
}
$$

Here $[n]=\{0,\ldots,n-1\}$, $\eta_S(c)=e_c$, and

$$
\operatorname{rank}_R\mathsf V_R(S)
=|\mathsf C(S)|
=|S|.
$$

Thus the entire descent is

$$
\boxed{
\begin{array}{c}
\text{profile / hierarchical shape}
\\
\downarrow\;\mathsf C
\\
\text{finite coordinate set}
\\
\downarrow\;R[-]
\\
\text{profiled finite free module}
\\
\downarrow\;\widetilde\ell_{S,D}
\\
\text{stride codomain }M.
\end{array}
}
$$

For the categorical Colfax construction, this supplies a literal functorial lift:

$$
\boxed{
\mathbf{Nest}
\xrightarrow{\;|\!-\!|\;}
\mathbf{FinSet}
\xrightarrow{\;R[-]\;}
\mathbf{FMod}_R.
}
$$

The first arrow is the realization functor of Carlisle--Shah--Stern--VanKoughnett; the second is the free-module functor. The composite sends each realized layout map to its basis-linear matrix.

The algebraic classification is therefore:

$$
\boxed{
\begin{aligned}
\text{flat tuples} &\rightsquigarrow \text{free associative monoid},\\
\text{linearized flat tuples} &\rightsquigarrow \text{tensor algebra},\\
\text{profiles} &\rightsquigarrow \text{non-symmetric operad},\\
\text{nested shapes} &\rightsquigarrow \text{profile-indexed tensor decompositions},\\
\text{layouts} &\rightsquigarrow \text{linear maps out of coordinate modules}.
\end{aligned}
}
$$

**Practical point.** A mode of extent $p$ becomes a based module $R^p$; a hierarchical shape becomes a profile-tagged tensor factorization; a stride becomes a linear map; and a proposed layout rewrite becomes a commuting diagram. This lets a layout designer move between hardware hierarchy, coordinate bases, sparse matrices, and executable CuTe shape-stride notation without erasing which modes were grouped together.

That is the short version. Everything below makes every object and every arrow precise.

---

## 0. Status and scope

**Convention 0.1.** The following statements have three different statuses.

1. The free-module adjunction, the isomorphism

   $$
   R[A\times B]\cong R[A]\otimes_RR[B],
   $$

   and monoidal coherence are standard theorems.

2. The profiled tensor realization defined here is a construction built from those theorems.

3. The claim that this construction is a useful semantics for CuTe is an interpretation. It is not asserted as a theorem of either cited CuTe paper.

**Scope 0.2.** This formulation exactly covers the in-bounds natural-coordinate and integral-coordinate semantics of layouts whose stride codomain is an $R$-module. In particular, it covers:

- integer-address layouts with $R=M=\mathbb Z$;
- rational or real-valued layouts;
- coordinate-valued layouts with $M=R^q$;
- binary linear layouts with $R=\mathbb F_2$.

Cris Cecka permits a more general integer-semimodule codomain with axioms weaker than those of an $R$-module. Standard semimodules over a commutative semiring have analogous tensor products, but extending the construction to exactly Cris's weaker axioms requires a separate choice of category. That extension is not silently assumed here.

Cris also defines an infinite extended domain of out-of-bounds congruent coordinates. The finite state module $R[\mathsf C(S)]$ in this note linearizes the in-bounds domain. To include the extended domain, one would replace $\mathsf C(S)$ by the corresponding congruent coordinate lattice; that extension is not used below.

**Boundary 0.3.** A field $K$ is indeed a one-dimensional vector space over itself, but vector spaces are only the field-valued special case of the module construction:

When $R=K$ is a field,

$$
\mathbf{Vect}_K=\mathbf{Mod}_K.
$$

For integer layouts, $\mathbb Z$-modules are the native choice.

**Boundary 0.4.** A bare module does not retain a CuTe profile. For example, all three modules

$$
(\mathbb Z^2\otimes\mathbb Z^3)\otimes\mathbb Z^4,
\qquad
\mathbb Z^2\otimes(\mathbb Z^3\otimes\mathbb Z^4),
\qquad
\mathbb Z^{24}
$$

are canonically isomorphic after bases and colexicographic factorization are fixed. The nesting resides in the chosen profile and its decomposition arrows, not in the isomorphism class of the final module.

---

## 1. Dictionary to Cris and Colfax

Let

- **Cris** mean Cris Cecka, *CuTe Layout Representation and Algebra*;
- **Colfax** mean Carlisle--Shah--Stern--VanKoughnett, *Categorical Foundations for CuTe Layouts*.

| This note | Cris | Colfax | Mathematical role here |
|---|---|---|---|
| finite tuple | Definition 2.1 | Definition 2.1.1.1 | a map from a finite index set |
| tuple concatenation | Section 3.1 for layouts | Definitions 2.1.1.3 and 2.1.3.36 | product of coordinate sets, hence tensor product after linearization |
| free tuple monoid | implicit in tuple operations | Remark 2.1.1.5 | flat words and the flat tensor algebra |
| hierarchical tuple / profile | Definitions 2.2--2.4 | Definitions 2.2.1.1 and 2.2.2.1 | syntax tree retained by the profiled tensor expression |
| shape and size | Definition 2.5 | Definitions 2.2.2.6 and 2.3.1.2 | coordinate-set cardinality and module rank |
| natural coordinate set | Definitions 2.6--2.11 | Notation 2.1.1.10 and Construction 2.1.2.13 | basis labels of the coordinate module |
| colexicographic conversion | equations (4)--(5) | Definition 2.1.2.18 | basis bijection between integral and natural coordinates |
| congruent stride | Definition 2.15 | Definition 2.3.1.1 | leafwise coefficients in a common module $M$ |
| layout function | Definition 2.17 and equation (7) | Constructions 2.1.2.19 and 2.3.1.17 | set map and its unique linear extension |
| permutation / transpose | coordinate reindexing throughout | Definitions 2.1.1.8 and 2.1.3.18 | symmetry or braiding isomorphism |
| flattening / reparenthesization | compatibility and coordinate conversion | Definition 2.3.2.1, Example 3.2.3.1, and Definition 3.2.4.1 | associator/unitor and basis reindexing |
| categorical realization | not the paper's chosen language | Definition 3.2.4.1 | functor $\mathbf{Nest}\to\mathbf{FinSet}$, then free-module lift |
| semi-linearity | Section 2.4.4 | encoded through coordinate and layout functions | explains exactly where the linear extension applies |

**Observation 1.1.** Cris begins with the implementation-level semantic object: a hierarchical shape and stride define a coordinate-to-codomain function.

**Observation 1.2.** Colfax factors a tractable part of the layout algebra through the categories $\mathbf{Tuple}$ and $\mathbf{Nest}$ and a realization functor into finite sets.

**Observation 1.3.** This note adds one layer:

$$
\mathbf{FinSet}\xrightarrow{R[-]}\mathbf{FMod}_R.
$$

It therefore does not replace either paper. It linearizes their coordinate semantics.

**Terminology warning 1.4.** Cris Definition 2.19 calls

$$
T=e\circ L
$$

a **tensor**, meaning an accessor composed with a layout. In this note, a **tensor product**

$$
V\otimes_RW
$$

is the algebraic monoidal product of modules. These are different uses of the word “tensor.” The construction here linearizes the coordinate space of a CuTe layout; it does not assert that a CuTe data tensor is a pure algebraic tensor.

**Rank warning 1.5.** CuTe rank counts top-level modes. Module rank counts basis elements. They are generally different:

$$
\operatorname{rank}_{\mathrm{CuTe}}(S)
=
\text{number of immediate modes},
$$

whereas

$$
\operatorname{rank}_R\mathsf V_R(S)=|S|.
$$

For example,

$$
\operatorname{rank}_{\mathrm{CuTe}}((2,3))=2,
\qquad
\operatorname{rank}_{\mathbb Z}\mathsf V_{\mathbb Z}(2,3)=6.
$$

---

## 2. Notation and arrows

**Convention 2.1.** All finite indices are zero-based:

$$
[n]:=\{0,1,\ldots,n-1\}.
$$

The empty product of sets is the singleton set $\mathbf 1=\{()\}$, the empty tensor product is $R$, the empty sum in a module is $0$, and the empty product of positive integers is $1$.

| Symbol | Name | Meaning |
|---|---|---|
| $A\to B$ | function arrow | an arbitrary function |
| $A\hookrightarrow B$ | injection | a one-to-one function |
| $A\twoheadrightarrow B$ | surjection | an onto function |
| $A\xrightarrow{\sim}B$ | isomorphism | an invertible structure-preserving map |
| $A\cong B$ | isomorphic | an isomorphism exists or has been canonically specified |
| $\bigsqcup$ | disjoint union / coproduct | elements retain which summand they came from |
| $\times$ | Cartesian product | pairs or tuples of choices |
| $\oplus$ | direct sum | finite-support linear combinations across summands |
| $\otimes_R$ | tensor product | universal recipient of $R$-bilinear maps |
| $\circ$ | composition | perform the right-hand map first |

**Convention 2.2.** The category of finite sets is $\mathbf{FinSet}$. The category of finite free $R$-modules is $\mathbf{FMod}_R$. Its symmetric monoidal structure is

$$
(\mathbf{FMod}_R,\otimes_R,R).
$$

The structural arrows are written

$$
\begin{aligned}
\alpha_{A,B,C}&:(A\otimes_RB)\otimes_RC\xrightarrow{\sim}A\otimes_R(B\otimes_RC),\\
\lambda_A&:R\otimes_RA\xrightarrow{\sim}A,\\
\rho_A&:A\otimes_RR\xrightarrow{\sim}A,\\
\beta_{A,B}&:A\otimes_RB\xrightarrow{\sim}B\otimes_RA.
\end{aligned}
$$

On pure tensors,

$$
\alpha((a\otimes b)\otimes c)=a\otimes(b\otimes c),
\qquad
\beta(a\otimes b)=b\otimes a.
$$

These are arrows, not definitional equalities.

---

## 3. Flat tuples, concatenation, and permutations

### 3.1 Tuples as maps

**Definition 3.1.** Let $X$ be a set. An $m$-tuple with entries in $X$ is a function

$$
x:[m]\longrightarrow X.
$$

Writing $x_i:=x(i)$ recovers the usual notation

$$
x=(x_0,\ldots,x_{m-1}).
$$

Thus

$$
X^m=\operatorname{Hom}_{\mathbf{Set}}([m],X).
$$

The set of all finite tuples is

$$
\boxed{
\operatorname{Tuple}(X)
:=
\bigsqcup_{m\ge 0}X^m.
}
$$

The disjoint union is necessary because tuples of different lengths are different sorts of data. Repeated entries are allowed because a function need not be injective.

### 3.2 Concatenation

**Definition 3.2.** The ordinal sum gives a canonical decomposition

$$
[m+n]=[m]\sqcup[n],
$$

where the second block is shifted by $m$. For

$$
x:[m]\to X,
\qquad
y:[n]\to X,
$$

their concatenation is the copaired map

$$
x\star y:[m+n]\longrightarrow X
$$

defined by

$$
(x\star y)(i)=
\begin{cases}
x(i),&0\le i<m,\\
y(i-m),&m\le i<m+n.
\end{cases}
$$

In list notation,

$$
(x_0,\ldots,x_{m-1})\star(y_0,\ldots,y_{n-1})
=
(x_0,\ldots,x_{m-1},y_0,\ldots,y_{n-1}).
$$

**Proposition 3.3.** The triple

$$
(\operatorname{Tuple}(X),\star,())
$$

is the free associative unital monoid on $X$.

**Proof.** Let $A$ be a monoid and let $f:X\to A$ be a function. Define

$$
\overline f(x_0,\ldots,x_{m-1})
:=
f(x_0)\cdots f(x_{m-1}),
$$

with $\overline f(())=1_A$. This is a monoid map, it extends $f$, and any monoid map extending $f$ must have this value on every word. Hence the extension exists uniquely. $\square$

The universal property is displayed by

$$
\begin{array}{ccc}
X&\xrightarrow{\iota}&\operatorname{Tuple}(X)\\
&\searrow f&\downarrow\exists!\,\overline f\\
&&A
\end{array}
\qquad
\overline f\circ\iota=f.
$$

This is precisely the content of Colfax Remark 2.1.1.5.

### 3.3 Permutations

**Definition 3.4.** The symmetric group on $m$ positions is

$$
\Sigma_m:=\operatorname{Aut}([m]).
$$

For $\sigma\in\Sigma_m$ and $x:[m]\to X$, define the right reindexing action

$$
x^\sigma:=x\circ\sigma.
$$

Equivalently,

$$
x^\sigma=(x_{\sigma(0)},\ldots,x_{\sigma(m-1)}).
$$

**Remark 3.5.** A tuple is not a special case of a permutation. A tuple labels positions by arbitrary elements of $X$; a permutation bijectively relabels the positions themselves:

$$
\begin{array}{ccc}
[m]&\xrightarrow[\sim]{\sigma}&[m]\\
&\searrow x^\sigma&\downarrow x\\
&&X.
\end{array}
$$

The triangle says $x^\sigma=x\circ\sigma$.

### 3.4 Flat linearization

Let

$$
R[X]:=\bigoplus_{x\in X}Re_x
$$

be the free $R$-module on $X$.

**Proposition 3.6.** There is a canonical isomorphism of associative unital $R$-algebras

$$
\boxed{
R[\operatorname{Tuple}(X)]
\cong
T_R(R[X])
:=
\bigoplus_{m\ge0}R[X]^{\otimes_Rm}.
}
$$

Under this isomorphism,

$$
e_{(x_0,\ldots,x_{m-1})}
\longmapsto
e_{x_0}\otimes\cdots\otimes e_{x_{m-1}},
$$

and concatenation of tuples becomes tensor-algebra multiplication.

**Proof.** In degree $m$, both sides are free on the same basis $X^m$:

$$
R[X^m]\xrightarrow{\sim}R[X]^{\otimes_Rm}.
$$

Taking the direct sum over $m$ gives the module isomorphism. The basis formula shows that it preserves the unit and concatenation. $\square$

**Boundary 3.7.** This proposition handles flat words. An ordinary associative algebra identifies

$$
(uv)w=u(vw).
$$

Consequently, an ordinary tensor algebra does not retain the distinction between the CuTe profiles

$$
((u,v),w),\qquad (u,(v,w)),\qquad (u,v,w).
$$

To retain those distinctions, the profile must remain part of the datum.

---

## 4. Profiles, hierarchical shapes, and coordinates

### 4.1 Profiles

**Definition 4.1.** The set $\mathsf{Profile}$ is recursively generated by

$$
P::=\ast\mid(P_0,\ldots,P_{r-1}),
\qquad r\ge0.
$$

The symbol $\ast$ is a leaf. A tuple of profiles is an ordered internal node. The empty profile $()$ is allowed.

The leaf count is defined by

$$
\operatorname{len}(\ast)=1,
\qquad
\operatorname{len}(P_0,\ldots,P_{r-1})
=
\sum_{j=0}^{r-1}\operatorname{len}(P_j).
$$

**Definition 4.2.** A hierarchical tuple over $X$ is either

$$
x\in X
$$

or a finite tuple

$$
(H_0,\ldots,H_{r-1})
$$

of hierarchical tuples over $X$.

Equivalently, as in Colfax Definition 2.2.2.1, it is a pair

$$
H=(H^\flat,\operatorname{prof}(H))
$$

consisting of a flat tuple $H^\flat\in X^m$ and a profile with $m$ leaves.

### 4.2 The profile operad

**Construction 4.3.** Put

$$
\mathsf{Profile}(m)
:=
\{P\in\mathsf{Profile}\mid\operatorname{len}(P)=m\}.
$$

If $P\in\mathsf{Profile}(m)$ and $Q_i\in\mathsf{Profile}(n_i)$, substitute $Q_i$ into the $i$th leaf of $P$. Denote the result by

$$
P[Q_0,\ldots,Q_{m-1}]
\in
\mathsf{Profile}(n_0+\cdots+n_{m-1}).
$$

Leaf substitution is unital and associative. Hence $\mathsf{Profile}$ is a non-symmetric operad, exactly as observed in Colfax Aside 2.2.1.10.

**Translation.** Flat concatenation is governed by a monoid. Arbitrary nested substitution is governed by an operad. This is why forcing the entire nested theory into one ordinary associative ring loses information.

### 4.3 Shapes

**Definition 4.4.** A hierarchical shape is a hierarchical tuple over $\mathbb N_{>0}$:

$$
S::=p\mid(S_0,\ldots,S_{r-1}),
\qquad p\in\mathbb N_{>0}.
$$

Its size is defined recursively by

$$
|p|:=p,
\qquad
|(S_0,\ldots,S_{r-1})|
:=
\prod_{j=0}^{r-1}|S_j|.
$$

Thus

$$
|()|=1.
$$

This is Cris Definition 2.5 and Colfax's size of a nested tuple.

### 4.4 Natural coordinate sets

**Definition 4.5.** The natural-coordinate set of a shape is defined recursively:

$$
\mathsf C(p):=[p],
$$

and

$$
\mathsf C(S_0,\ldots,S_{r-1})
:=
\prod_{j=0}^{r-1}\mathsf C(S_j).
$$

In particular,

$$
\mathsf C(())=\mathbf1.
$$

An element of $\mathsf C(S)$ has exactly the same profile as $S$.

**Proposition 4.6.** For every hierarchical shape $S$,

$$
\boxed{
|\mathsf C(S)|=|S|.
}
$$

**Proof.** By structural induction. For a leaf $p$,

$$
|\mathsf C(p)|=|[p]|=p=|p|.
$$

For a tuple node,

$$
\begin{aligned}
|\mathsf C(S_0,\ldots,S_{r-1})|
&=
\left|\prod_j\mathsf C(S_j)\right|\\
&=
\prod_j|\mathsf C(S_j)|\\
&=
\prod_j|S_j|\\
&=
|(S_0,\ldots,S_{r-1})|.
\end{aligned}
$$

The empty case follows from the empty-product convention. $\square$

### 4.5 Integral and natural coordinates

**Construction 4.7.** Define mutually inverse colexicographic maps

$$
\operatorname{idx2crd}_S:[|S|]\xrightarrow{\sim}\mathsf C(S)
$$

and

$$
\operatorname{crd2idx}_S:\mathsf C(S)\xrightarrow{\sim}[|S|]
$$

recursively.

For a leaf $p$, both maps are the identity of $[p]$.

For the empty node, they are the unique mutually inverse maps

$$
[1]\xrightleftarrows[\operatorname{crd2idx}_{()}]{\operatorname{idx2crd}_{()}}\mathbf1.
$$

For

$$
S=(S_0,\ldots,S_{r-1}),
\qquad
N_j:=|S_j|,
$$

put

$$
q_j(i)
:=
\left\lfloor
\frac{i}{\prod_{k<j}N_k}
\right\rfloor
\bmod N_j.
$$

Then

$$
\operatorname{idx2crd}_S(i)
:=
\left(
\operatorname{idx2crd}_{S_0}(q_0(i)),
\ldots,
\operatorname{idx2crd}_{S_{r-1}}(q_{r-1}(i))
\right).
$$

Conversely,

$$
\operatorname{crd2idx}_S(c_0,\ldots,c_{r-1})
:=
\sum_{j=0}^{r-1}
\operatorname{crd2idx}_{S_j}(c_j)
\prod_{k<j}N_k.
$$

These are Cris's $\operatorname{idx2crd}$ and $\operatorname{crd2idx}$, recursively applied, and Colfax's colexicographic isomorphism.

They are mutually inverse by uniqueness of the mixed-radix expansion

$$
i=\sum_jq_j(i)\prod_{k<j}N_k,
\qquad
0\le q_j(i)<N_j.
$$

**Example 4.8.** For

$$
S=((2,3),2),
$$

the integral coordinate $9$ becomes

$$
\operatorname{idx2crd}_S(9)=((1,1),1),
$$

because

$$
9=1+2\cdot1+6\cdot1.
$$

### 4.6 Flattening and refinement

**Definition 4.9.** The flattening $S^\flat$ is the ordered tuple of leaf extents of $S$.

There is a canonical profile-forgetting bijection

$$
\operatorname{flat}_S:
\mathsf C(S)\xrightarrow{\sim}\mathsf C(S^\flat)
$$

obtained by deleting coordinate parentheses while preserving leaf order.

If $S$ and $T$ have the same ordered leaf extents, their reparenthesization bijection is

$$
r_{S,T}
:=
\operatorname{flat}_T^{-1}\circ\operatorname{flat}_S:
\mathsf C(S)\xrightarrow{\sim}\mathsf C(T).
$$

If a leaf $pq$ is refined into $(p,q)$, the colexicographic refinement bijection is

$$
\delta_{p,q}:[pq]\xrightarrow{\sim}[p]\times[q],
\qquad
n\longmapsto
\left(n\bmod p,\left\lfloor\frac np\right\rfloor\right).
$$

**Remark 4.10.** The isomorphism

$$
R^{pq}\cong R^p\otimes_RR^q
$$

is not determined by the number $pq$ alone. It becomes canonical after the ordered factorization $(p,q)$ and the colex basis convention are supplied. The CuTe refinement supplies exactly that missing structure.

---

## 5. The free-module engine

### 5.1 Free modules

**Definition 5.1.** For a set $A$, define

$$
R[A]
:=
\bigoplus_{a\in A}Re_a.
$$

An element is a finite formal sum

$$
\sum_{a\in A}r_ae_a.
$$

**Theorem 5.2 (free-module universal property).** For every set $A$ and every $R$-module $M$, restriction to basis elements defines a natural bijection

$$
\boxed{
\operatorname{Hom}_R(R[A],M)
\xrightarrow{\sim}
\operatorname{Hom}_{\mathbf{Set}}(A,U(M)).
}
$$

The inverse sends a function $f:A\to U(M)$ to

$$
\overline f\left(\sum_ar_ae_a\right)
:=
\sum_ar_af(a).
$$

Equivalently,

$$
R[-]\dashv U.
$$

The universal triangle is

$$
\begin{array}{ccc}
A&\xrightarrow{\eta_A}&U(R[A])\\
&\searrow f&\downarrow U(\overline f)\\
&&U(M),
\end{array}
\qquad
\eta_A(a)=e_a.
$$

### 5.2 Cartesian products become tensor products

**Theorem 5.3 (strong symmetric monoidality).** For sets $A$ and $B$, the map

$$
\chi_{A,B}:R[A\times B]\longrightarrow R[A]\otimes_RR[B]
$$

defined by

$$
\chi_{A,B}(e_{(a,b)})=e_a\otimes e_b
$$

is a natural isomorphism. The unit comparison is

$$
\chi_0:R[\mathbf1]\xrightarrow{\sim}R,
\qquad
e_{()}\longmapsto1.
$$

These maps make

$$
R[-]:(\mathbf{Set},\times,\mathbf1)
\longrightarrow
(\mathbf{Mod}_R,\otimes_R,R)
$$

a strong symmetric monoidal functor.

**Proof.** The module $R[A\times B]$ has basis indexed by pairs $(a,b)$. The tensor product $R[A]\otimes_RR[B]$ has basis $e_a\otimes e_b$, also indexed by pairs. The stated basis map is therefore an isomorphism. Naturality, compatibility with the unit, associator, and symmetry can all be checked on a basis vector. Every route sends a basis vector to the same pure tensor. $\square$

For a finite family,

$$
\boxed{
R\left[\prod_{j=0}^{r-1}A_j\right]
\xrightarrow{\sim}
\bigotimes_{j=0}^{r-1}R[A_j].
}
$$

This theorem is the exact reason tensor products appear. They do not replace Cartesian products; they are what Cartesian products of basis sets become after free linearization.

---

## 6. Profiled tensor modules

### 6.1 Formal tensor expressions

**Definition 6.1.** For each shape $S$, define a formal tensor expression $\mathsf E_R(S)$ recursively:

$$
\mathsf E_R(p):=R^{([p])}
:=
\bigoplus_{i\in[p]}Re_i,
$$

and

$$
\mathsf E_R(S_0,\ldots,S_{r-1})
:=
\left\langle
\mathsf E_R(S_0)\otimes_R\cdots\otimes_R\mathsf E_R(S_{r-1})
\right\rangle.
$$

The angle brackets retain the arity and parenthesization of the source node. For the empty node,

$$
\mathsf E_R(())=\langle R\rangle.
$$

Let

$$
\operatorname{ev}_R(\mathsf E_R(S))
$$

denote its evaluation in $\mathbf{FMod}_R$. Recursively evaluate the children and use the left-associated $r$-fold tensor product at an $r$-ary node, with $R$ at a zero-ary node. The angle-bracketed syntax remains part of the datum even when two expressions evaluate to the same chosen module.

### 6.2 The realization

**Definition 6.2.** The coordinate module of $S$ is

$$
\boxed{
\mathsf V_R(S):=R[\mathsf C(S)].
}
$$

Write $\operatorname{Node}(S)$ for the set of internal tuple nodes of the shape tree. For $v\in\operatorname{Node}(S)$, write $S|_v$ for the subtree rooted at $v$.

The profiled tensor realization of $S$ is the datum

$$
\boxed{
\mathbb T_R(S)
:=
\left(
\operatorname{prof}(S),
\mathsf V_R(S),
\{\chi_{S|_v}\}_{v\in\operatorname{Node}(S)}
\right),
}
$$

where, at each internal node

$$
S|_v=(S_0,\ldots,S_{r-1}),
$$

the structural map is

$$
\chi_{S|_v}:
\mathsf V_R(S|_v)
\xrightarrow{\sim}
\bigotimes_{j=0}^{r-1}\mathsf V_R(S_j).
$$

**Theorem 6.3 (profiled tensor realization).** For every hierarchical shape $S$, there is a canonical isomorphism

$$
\Phi_S:
\mathsf V_R(S)
\xrightarrow{\sim}
\operatorname{ev}_R(\mathsf E_R(S))
$$

characterized recursively by

$$
\Phi_p(e_i)=e_i
$$

at a leaf and

$$
\Phi_{(S_0,\ldots,S_{r-1})}(e_{(c_0,\ldots,c_{r-1})})
=
\Phi_{S_0}(e_{c_0})\otimes\cdots\otimes\Phi_{S_{r-1}}(e_{c_{r-1}})
$$

at a tuple node.

Moreover,

$$
\boxed{
\operatorname{rank}_R\mathsf V_R(S)=|S|.
}
$$

**Proof.** The isomorphism is obtained recursively from Theorem 5.3. The basis formula proves bijectivity at each node. The rank statement follows from Proposition 4.6:

$$
\operatorname{rank}_RR[\mathsf C(S)]
=|\mathsf C(S)|
=|S|.
$$

$\square$

### 6.3 Substitution

Let $P$ be a profile with $m$ leaves and let $S_0,\ldots,S_{m-1}$ be shapes. Write

$$
P[S_0,\ldots,S_{m-1}]
$$

for the hierarchical shape obtained by substituting $S_i$ into the $i$th leaf of $P$.

Write

$$
\bigotimes\nolimits_P\mathsf V_R(S_i)
$$

for the formal tensor expression obtained by placing the ordered factors $\mathsf V_R(S_i)$ into the leaves of $P$ and evaluating each internal node by tensor product.

**Proposition 6.4.** There is a canonical profile-shaped tensor isomorphism

$$
\boxed{
\Gamma_{P;S_0,\ldots,S_{m-1}}:
\bigotimes\nolimits_P
\mathsf V_R(S_i)
\xrightarrow{\sim}
\mathsf V_R(P[S_0,\ldots,S_{m-1}]).
}
$$

These maps satisfy the operadic unit and associativity diagrams.

**Proof.** Both sides have a basis indexed by the same profile-shaped tuple of coordinates. Define $\Gamma$ by sending the pure tensor of those basis elements to the basis element indexed by the corresponding substituted coordinate. Every operadic unit or associativity diagram sends a pure basis tensor to that same final basis element, so every such diagram commutes. $\square$

**Conclusion 6.5.** The family

$$
\{\mathsf V_R(S)\}_{S\in\mathsf{HShape}}
$$

is naturally profile-operadic. If one forms the large direct sum

$$
\mathscr V_R
:=
\bigoplus_S\mathsf V_R(S),
$$

the grafting operations land in profile-specific summands. They are not an ordinary associative ring multiplication:

$$
\mathsf V_R((S,T),U)
\ne
\mathsf V_R(S,(T,U))
$$

as profile-graded summands, although a canonical associator gives

$$
\mathsf V_R((S,T),U)
\xrightarrow{\sim}
\mathsf V_R(S,(T,U)).
$$

This is the clean version of “a ring of vectors that includes nesting”: retain a profile-graded module with operadic tensor operations. Quotienting the associator arrows into literal equalities recovers a flatter associative algebra and discards the nesting distinction.

### 6.4 Coherence and reparenthesization

**Theorem 6.6.** If $S$ and $T$ have the same ordered leaf extents, then the basis map

$$
R[r_{S,T}]:
\mathsf V_R(S)\xrightarrow{\sim}\mathsf V_R(T)
$$

agrees, under $\Phi_S$ and $\Phi_T$, with the canonical composite of associators and unitors between their formal tensor expressions.

**Proof.** Both maps send the basis element indexed by a nested coordinate of $S$ to the pure tensor with the same ordered leaf coordinates, then regard that tensor with profile $T$. Mac Lane coherence guarantees that all composites of canonical associators and unitors with these endpoints agree. $\square$

For three factors, the fundamental square is

$$
\begin{array}{ccc}
R[(A\times B)\times C]
&\xrightarrow{\chi}&
(R[A]\otimes_RR[B])\otimes_RR[C]
\\
\downarrow R[\alpha^\times]&&\downarrow\alpha
\\
R[A\times(B\times C)]
&\xrightarrow{\chi}&
R[A]\otimes_R(R[B]\otimes_RR[C]),
\end{array}
$$

and it commutes on every basis vector.

---

## 7. Strides and layouts as linear maps

### 7.1 Congruent strides

**Definition 7.1.** Let $M$ be an $R$-module. A stride $D$ for a shape $S$ is a hierarchical tuple of elements of $M$ with the same profile as $S$.

Equivalently, if $\operatorname{Leaf}(S)$ is the ordered set of leaves of $S$, then

$$
D=(d_\lambda)_{\lambda\in\operatorname{Leaf}(S)},
\qquad
d_\lambda\in M,
$$

together with the profile of $S$.

This is the module-valued specialization of Cris Definition 2.15 and the congruence condition in Colfax Definition 2.3.1.1.

### 7.2 Natural-coordinate layout

**Definition 7.2.** For $c\in\mathsf C(S)$, write $c_\lambda\in\mathbb N$ for its coordinate at leaf $\lambda$. Let

$$
[c_\lambda]_R:=c_\lambda 1_R\in R.
$$

Define

$$
\boxed{
\ell_{S,D}:\mathsf C(S)\longrightarrow U(M),
\qquad
\ell_{S,D}(c)
:=
\sum_{\lambda\in\operatorname{Leaf}(S)}
[c_\lambda]_R\,d_\lambda.
}
$$

For $R=M=\mathbb Z$, this is exactly the usual shape-stride inner product.

For the empty layout,

$$
\ell_{(),()}(())=0.
$$

### 7.3 Component module versus state module

There are two useful linear objects, and they must not be conflated.

**Definition 7.3.** The leaf-component module is

$$
\mathsf Q_R(S)
:=
R[\operatorname{Leaf}(S)]
=
\bigoplus_{\lambda\in\operatorname{Leaf}(S)}Re_\lambda.
$$

Its rank is the number of leaf coordinates:

$$
\operatorname{rank}_R\mathsf Q_R(S)
=
\operatorname{len}(S).
$$

The coordinate-state module remains

$$
\mathsf V_R(S)=R[\mathsf C(S)],
$$

whose rank is the number of valid coordinate states:

$$
\operatorname{rank}_R\mathsf V_R(S)=|S|.
$$

At a tuple node, the two constructions obey different laws:

$$
\boxed{
\begin{aligned}
\mathsf Q_R(S_0,\ldots,S_{r-1})
&\cong
\bigoplus_j\mathsf Q_R(S_j),
\\
\mathsf V_R(S_0,\ldots,S_{r-1})
&\cong
\bigotimes_j\mathsf V_R(S_j).
\end{aligned}
}
$$

The first law comes from disjoint union of leaves. The second comes from Cartesian product of coordinate choices.

Indeed,

$$
\operatorname{Leaf}(S_0,\ldots,S_{r-1})
\cong
\bigsqcup_j\operatorname{Leaf}(S_j),
$$

so

$$
R\left[\bigsqcup_j\operatorname{Leaf}(S_j)\right]
\cong
\bigoplus_jR[\operatorname{Leaf}(S_j)].
$$

Likewise,

$$
\mathsf C(S_0,\ldots,S_{r-1})
=
\prod_j\mathsf C(S_j),
$$

so Theorem 5.3 supplies the tensor isomorphism.

**Construction 7.4.** Define the coordinate-vector embedding

$$
j_S:\mathsf C(S)\longrightarrow U\mathsf Q_R(S)
$$

by

$$
j_S(c)
:=
\sum_\lambda[c_\lambda]_Re_\lambda.
$$

The stride defines an $R$-linear map

$$
d_D:\mathsf Q_R(S)\longrightarrow M,
\qquad
d_D(e_\lambda)=d_\lambda.
$$

Then

$$
\boxed{
\ell_{S,D}=U(d_D)\circ j_S.
}
$$

Linearizing $j_S$ gives

$$
\widetilde j_S:\mathsf V_R(S)\longrightarrow\mathsf Q_R(S),
\qquad
\widetilde j_S(e_c)=j_S(c),
$$

and therefore

$$
\boxed{
\widetilde\ell_{S,D}
=
d_D\circ\widetilde j_S.
}
$$

This is the exact connection to Cris's linear form $D\widetilde c$: the map $d_D$ is linear on the component module $\mathsf Q_R(S)$, while the tensor-factorized module $\mathsf V_R(S)$ linearizes the finite set of complete coordinate states.

### 7.4 Unique linear extension

**Construction 7.5.** By Theorem 5.2, $\ell_{S,D}$ extends uniquely to

$$
\boxed{
\widetilde\ell_{S,D}:
\mathsf V_R(S)\longrightarrow M
}
$$

such that

$$
\widetilde\ell_{S,D}(e_c)=\ell_{S,D}(c).
$$

The defining diagram is

$$
\begin{array}{ccc}
\mathsf C(S)&\xrightarrow{\eta_S}&U\mathsf V_R(S)\\
&\searrow\ell_{S,D}&\downarrow U\widetilde\ell_{S,D}\\
&&U(M).
\end{array}
$$

**Proposition 7.6.** The extension is

$$
\widetilde\ell_{S,D}
\left(
\sum_{c\in\mathsf C(S)}a_ce_c
\right)
=
\sum_{c\in\mathsf C(S)}a_c\,\ell_{S,D}(c).
$$

In particular, when $M=R$, it is a linear functional

$$
\widetilde\ell_{S,D}\in\mathsf V_R(S)^\vee.
$$

### 7.5 Recursive tensor formula

**Definition 7.7.** Define the augmentation

$$
\varepsilon_S:\mathsf V_R(S)\longrightarrow R,
\qquad
\varepsilon_S(e_c)=1.
$$

At a leaf $p:d$, define

$$
\lambda_{p,d}:R^{([p])}\longrightarrow M,
\qquad
e_i\longmapsto[i]_R\,d.
$$

**Proposition 7.8.** Suppose

$$
S=(S_0,\ldots,S_{r-1}),
\qquad
D=(D_0,\ldots,D_{r-1}).
$$

The layout extension is

$$
\boxed{
\widetilde\ell_{S,D}
=
\left(
\sum_{j=0}^{r-1}
\varepsilon_{S_0}\otimes\cdots\otimes
\widetilde\ell_{S_j,D_j}
\otimes\cdots\otimes\varepsilon_{S_{r-1}}
\right)
\circ\chi_S
}
$$

where the canonical unitors identify

$$
R\otimes\cdots\otimes M\otimes\cdots\otimes R
\cong M.
$$

**Proof.** On a pure basis tensor,

$$
e_{c_0}\otimes\cdots\otimes e_{c_{r-1}},
$$

the $j$th summand evaluates to

$$
\ell_{S_j,D_j}(c_j),
$$

because every other augmentation returns $1$. Summing over $j$ gives

$$
\sum_j\ell_{S_j,D_j}(c_j)
=
\ell_{S,D}(c_0,\ldots,c_{r-1}).
$$

Uniqueness follows from the free-module universal property. $\square$

This formula is the precise relation between tensor structure and concatenated address arithmetic:

$$
\text{coordinate factors combine by }\otimes_R,
\qquad
\text{their address contributions combine by }+_M.
$$

Tensor product is therefore not being mislabeled as numeric concatenation.

### 7.6 Integral layout function

**Definition 7.9.** The integral layout function is

$$
L_{S,D}
:=
\ell_{S,D}\circ\operatorname{idx2crd}_S:
[|S|]\longrightarrow U(M).
$$

Equivalently,

$$
\boxed{
L_{S,D}
=
U(\widetilde\ell_{S,D})
\circ\eta_S
\circ\operatorname{idx2crd}_S.
}
$$

For a flat integer layout, this is Colfax Construction 2.1.2.19. For a nested layout, Colfax defines the layout function by flattening; Theorem 6.6 shows that this agrees with the profiled tensor realization.

**Observation 7.10.** Cris writes a layout as

$$
L=D\circ S.
$$

In the present notation, the shape part converts an allowed coordinate to a natural coordinate, while the stride part is $\ell_{S,D}$. On natural coordinates the stride part is linear; the finite coordinate set itself is not closed under module addition. This is exactly Cris's semi-linearity distinction in Section 2.4.4.

---

## 8. Operations and their tensor meanings

### 8.1 Flat concatenation versus nested concatenation

Let

$$
S=(S_0,\ldots,S_{m-1}),
\qquad
T=(T_0,\ldots,T_{n-1})
$$

be top-level tuple shapes.

Their flat splice is

$$
S\star T
=
(S_0,\ldots,S_{m-1},T_0,\ldots,T_{n-1}).
$$

Their nested two-fold concatenation is

$$
(S,T).
$$

These are different profiles, as emphasized in Colfax Remarks 2.3.2.6 and 2.3.2.9. Both have coordinate sets canonically bijective to

$$
\mathsf C(S)\times\mathsf C(T),
$$

so both have coordinate modules canonically isomorphic to

$$
\mathsf V_R(S)\otimes_R\mathsf V_R(T).
$$

The difference is retained by the profile tag.

### 8.2 Concatenated layouts

Let

$$
f:\mathsf V_R(S)\to M,
\qquad
g:\mathsf V_R(T)\to M.
$$

Define

$$
f\boxplus g:
\mathsf V_R((S,T))\longrightarrow M
$$

by

$$
\boxed{
f\boxplus g
:=
\left(
(f\otimes\varepsilon_T)
+
(\varepsilon_S\otimes g)
\right)
\circ\chi_{(S,T)}.
}
$$

with unitors used to identify each codomain with $M$.

On basis elements, equivalently on their pure-tensor images,

$$
(f\boxplus g)(e_{(c,c')})
=
f(e_c)+g(e_{c'}).
$$

**Proposition 8.1.** If $D$ and $E$ are strides for $S$ and $T$, then

$$
\boxed{
\widetilde\ell_{(S,T),(D,E)}
=
\widetilde\ell_{S,D}\boxplus
\widetilde\ell_{T,E}.
}
$$

This is the tensor-linear form of Cris equation (11) and Colfax Proposition 2.1.3.40.

### 8.3 Permutations

Let

$$
S=(S_0,\ldots,S_{m-1})
$$

and $\sigma\in\Sigma_m$. Define

$$
S^\sigma:=(S_{\sigma(0)},\ldots,S_{\sigma(m-1)}).
$$

There is a coordinate bijection

$$
p_\sigma:\mathsf C(S)\xrightarrow{\sim}\mathsf C(S^\sigma),
\qquad
(c_0,\ldots,c_{m-1})
\longmapsto
(c_{\sigma(0)},\ldots,c_{\sigma(m-1)}).
$$

Linearization gives

$$
R[p_\sigma]:
\mathsf V_R(S)\xrightarrow{\sim}\mathsf V_R(S^\sigma).
$$

Under the tensor decompositions, this is the symmetry map

$$
\beta_\sigma:
\bigotimes_{i=0}^{m-1}\mathsf V_R(S_i)
\xrightarrow{\sim}
\bigotimes_{i=0}^{m-1}\mathsf V_R(S_{\sigma(i)}).
$$

**Proposition 8.2.** If the stride and coordinate are reindexed with the shape, then the address is unchanged:

$$
\boxed{
\ell_{S^\sigma,D^\sigma}(p_\sigma(c))
=
\ell_{S,D}(c).
}
$$

Equivalently, the square

$$
\begin{array}{ccc}
\mathsf V_R(S)&\xrightarrow{R[p_\sigma]}&\mathsf V_R(S^\sigma)\\
\downarrow\widetilde\ell_{S,D}&&\downarrow\widetilde\ell_{S^\sigma,D^\sigma}\\
M&=&M
\end{array}
$$

commutes.

**Proof.** Both sides are the same finite sum with its terms reordered:

$$
\sum_i[c_i]_Rd_i.
$$

$\square$

### 8.4 Reparenthesization and flattening

Suppose layouts $S:D$ and $T:E$ have the same ordered leaf extents and the same ordered leaf strides. Then

$$
\boxed{
\ell_{T,E}\circ r_{S,T}
=
\ell_{S,D}.
}
$$

Consequently,

$$
\begin{array}{ccc}
\mathsf V_R(S)&\xrightarrow{R[r_{S,T}]}&\mathsf V_R(T)\\
\downarrow\widetilde\ell_{S,D}&&\downarrow\widetilde\ell_{T,E}\\
M&=&M
\end{array}
$$

commutes.

This is the tensor-module realization of Colfax reparenthesization isomorphisms. It also explains why Colfax nested layouts with the same flattening have the same integral layout function.

### 8.5 Coarsening and refinement

Replacing a leaf $pq$ by the nested mode $(p,q)$ produces the diagram

$$
\begin{array}{ccc}
R^{pq}&\xrightarrow[\cong]{R[\delta_{p,q}]}&R^p\otimes_RR^q\\
\downarrow\widetilde\ell&&\downarrow\widetilde\ell'\\
M&=&M.
\end{array}
$$

The diagram commutes exactly when the refined stride represents the same address function. For contiguous colex refinement, a leaf stride $d$ refines to

$$
(d,pd),
$$

because

$$
n\,d
=
(n\bmod p)d
+
\left\lfloor\frac np\right\rfloor(pd).
$$

This is the smallest example of tensor factorization generating a nested layout.

### 8.6 Coalesce

If a coalesce operation replaces $S:D$ by $T:E$ while preserving the integral layout function, then

$$
L_{S,D}=L_{T,E}:[|S|]\to M
$$

and $|S|=|T|$. Therefore the colex basis isomorphism

$$
q_{S,T}
:=
\operatorname{idx2crd}_T
\circ
\operatorname{crd2idx}_S
:
\mathsf C(S)\xrightarrow{\sim}\mathsf C(T)
$$

satisfies

$$
\widetilde\ell_{T,E}\circ R[q_{S,T}]
=
\widetilde\ell_{S,D}.
$$

The tensor formulation records the semantic equality as a commuting triangle. It does not determine when CuTe's coalesce algorithm is legal or minimal; those conditions remain the work of the source papers.

### 8.7 Two different linearizations

This distinction is essential.

Given a set map

$$
f:A\to B,
$$

the free-module functor produces the basis-preserving linear map

$$
R[f]:R[A]\to R[B],
\qquad
e_a\mapsto e_{f(a)}.
$$

This construction is functorial:

$$
\boxed{
R[g\circ f]=R[g]\circ R[f].
}
$$

If the codomain is the underlying set of an $R$-module $M$, the numerical linear extension is instead

$$
\widetilde f:R[A]\to M,
\qquad
e_a\mapsto f(a).
$$

It factors as

$$
\boxed{
R[A]
\xrightarrow{R[f]}
R[U(M)]
\xrightarrow{\epsilon_M}
M,
}
$$

where

$$
\epsilon_M(e_m)=m
$$

is the counit of the free-forgetful adjunction.

**Warning 8.3.** The map $R[f]$ always preserves composition. The collapsed numerical map $\widetilde f$ preserves postcomposition only when the postcomposing map is $R$-linear:

$$
\widetilde{h\circ f}=h\circ\widetilde f
\qquad
\text{if }h:M\to N\text{ is }R\text{-linear}.
$$

An arbitrary CuTe layout is only semi-linear because coordinate decoding can be nonlinear. Therefore one cannot claim that every CuTe composition becomes ordinary composition of the numerical functionals $\widetilde\ell$.

### 8.8 Exact lift of the Colfax realization

Colfax Definition 3.2.4.1 gives a functor

$$
|\!-\!|:\mathbf{Nest}\to\mathbf{FinSet}.
$$

Define

$$
\boxed{
\mathcal L_R
:=
R[-]\circ|\!-\!|
:
\mathbf{Nest}\longrightarrow\mathbf{FMod}_R.
}
$$

Then

$$
\mathcal L_R(g\circ f)
=
\mathcal L_R(g)\circ\mathcal L_R(f)
$$

by functoriality. In chosen bases, $\mathcal L_R(f)$ is the matrix with

$$
\mathcal L_R(f)e_i=e_{|f|(i)}.
$$

Thus every column has exactly one entry equal to $1$ and every other entry equal to $0$. Isomorphisms, including permutations and reparenthesizations, become permutation matrices.

For an object $S$ of $\mathbf{Nest}$, Colfax's realized finite set is the integral-coordinate set $[|S|]$. The colex map gives a canonical comparison

$$
\Theta_S
:=
R[\operatorname{idx2crd}_S]
:
R^{([|S|])}
\xrightarrow{\sim}
\mathsf V_R(S).
$$

Thus the same functor, expressed in natural-coordinate bases, sends $f:S\to T$ to

$$
\boxed{
\widehat{\mathcal L}_R(f)
:=
\Theta_T\circ\mathcal L_R(f)\circ\Theta_S^{-1}
:
\mathsf V_R(S)\longrightarrow\mathsf V_R(T).
}
$$

Because this is objectwise conjugation of a functor,

$$
\widehat{\mathcal L}_R(g\circ f)
=
\widehat{\mathcal L}_R(g)\circ\widehat{\mathcal L}_R(f).
$$

This is an exact theorem, not an analogy.

### 8.9 Operation-status table

| CuTe / Colfax operation | Tensor-module translation | Status |
|---|---|---|
| tuple or nested concatenation | Cartesian product of coordinate bases, then $\otimes_R$ | exact canonical isomorphism |
| address contribution of concatenation | $\boxplus$ using augmentations | exact |
| permutation / transpose | symmetry map $\beta_\sigma$ | exact |
| reparenthesization / flattening | associator and unitor maps | exact |
| compatible refinement | chosen mixed-radix basis isomorphism | exact after the colex convention |
| fixed layout evaluation | unique linear extension $\widetilde\ell$ | exact |
| Colfax morphism composition | composition of $R[|f|]$ | exact |
| coalesce preserving layout function | commuting basis-change triangle | exact semantically; legality is external |
| arbitrary CuTe composition | basis-linearization remains functorial; collapsed functional need not | conditional |
| complement, logical division, logical product | candidates for additional diagrams and universal constructions | not derived here |
| arbitrary Cris integer-semimodule | requires semimodule tensor products | outside the present module scope |

---

## 9. Worked examples

### Example 9.1: the empty layout

Let

$$
S=(),
\qquad
D=().
$$

Then

$$
\mathsf C(S)=\{()\},
\qquad
\mathsf V_R(S)=R[\{()\}]\cong R,
\qquad
|S|=1.
$$

The layout extension is the zero map

$$
\widetilde\ell_{(),()}:R\to M,
\qquad
1\mapsto0.
$$

The empty shape is therefore represented by the tensor unit, not the zero module.

### Example 9.2: a flat Colfax layout

Take the flat integer layout

$$
L=(2,3):(1,5).
$$

Its natural-coordinate set is

$$
\mathsf C(2,3)=[2]\times[3].
$$

Its coordinate module is

$$
\mathsf V_{\mathbb Z}(2,3)
=
\mathbb Z[\mathsf C(2,3)]
\xrightarrow{\sim}
\mathbb Z^2\otimes_{\mathbb Z}\mathbb Z^3
\cong
\mathbb Z^6.
$$

Its leaf-component module is only

$$
\mathsf Q_{\mathbb Z}(2,3)\cong\mathbb Z^2.
$$

The two stages are

$$
j(i,j)=ie_0+je_1,
\qquad
d_D(ae_0+be_1)=a+5b.
$$

Hence

$$
\ell=d_D\circ j.
$$

The natural-coordinate function is

$$
\ell(i,j)=i+5j.
$$

The integral layout values in colex order are

$$
\begin{array}{c|cccccc}
n&0&1&2&3&4&5\\
\hline
\operatorname{idx2crd}(n)
&(0,0)&(1,0)&(0,1)&(1,1)&(0,2)&(1,2)\\
L(n)&0&1&5&6&10&11.
\end{array}
$$

The linear extension is

$$
\widetilde\ell:\mathbb Z^6\to\mathbb Z
$$

with

$$
\widetilde\ell(a_0,\ldots,a_5)
=
0a_0+1a_1+5a_2+6a_3+10a_4+11a_5.
$$

This does not say that a coordinate is an arbitrary vector in $\mathbb Z^6$. It says that each valid coordinate names a basis vector and the layout extends linearly from those basis values.

### Example 9.3: nested contiguous layout

Take

$$
S=((2,3),4),
\qquad
D=((1,2),6).
$$

Then

$$
\mathsf C(S)=([2]\times[3])\times[4]
$$

and

$$
\mathsf V_{\mathbb Z}(S)
\xrightarrow{\sim}
(\mathbb Z^2\otimes\mathbb Z^3)\otimes\mathbb Z^4.
$$

The layout is

$$
\ell((i,j),k)=i+2j+6k.
$$

For example,

$$
\ell((1,2),3)=1+2\cdot2+6\cdot3=23.
$$

The module rank is

$$
2\cdot3\cdot4=24.
$$

On a basis tensor,

$$
\widetilde\ell
\bigl((e_i\otimes e_j)\otimes e_k\bigr)
=
i+2j+6k.
$$

The recursive formula is

$$
\widetilde\ell
=
\lambda_{2,1}\otimes\varepsilon_3\otimes\varepsilon_4
+
\varepsilon_2\otimes\lambda_{3,2}\otimes\varepsilon_4
+
\varepsilon_2\otimes\varepsilon_3\otimes\lambda_{4,6},
$$

with the displayed parenthesization understood through the structural arrows.

### Example 9.4: same leaves, different profile

Reparenthesize the previous layout:

$$
T=(2,(3,4)),
\qquad
E=(1,(2,6)).
$$

Now

$$
\mathsf V_{\mathbb Z}(T)
\xrightarrow{\sim}
\mathbb Z^2\otimes(\mathbb Z^3\otimes\mathbb Z^4).
$$

The associator

$$
\alpha:
(\mathbb Z^2\otimes\mathbb Z^3)\otimes\mathbb Z^4
\xrightarrow{\sim}
\mathbb Z^2\otimes(\mathbb Z^3\otimes\mathbb Z^4)
$$

sends

$$
(e_i\otimes e_j)\otimes e_k
\longmapsto
e_i\otimes(e_j\otimes e_k).
$$

The layout diagram commutes:

$$
\begin{array}{ccc}
(\mathbb Z^2\otimes\mathbb Z^3)\otimes\mathbb Z^4
&\xrightarrow{\alpha}&
\mathbb Z^2\otimes(\mathbb Z^3\otimes\mathbb Z^4)
\\
\downarrow\widetilde\ell_{S,D}
&&
\downarrow\widetilde\ell_{T,E}
\\
\mathbb Z&=&\mathbb Z.
\end{array}
$$

The two layouts are not the same profiled object. They have the same flattened coordinate semantics.

### Example 9.5: refining one mode

Start with the one-dimensional contiguous shape

$$
6:1.
$$

Refine $6=2\cdot3$:

$$
6:1
\rightsquigarrow
(2,3):(1,2).
$$

The coordinate bijection is

$$
[6]\xrightarrow{\sim}[2]\times[3],
\qquad
n\longmapsto
\left(n\bmod2,\left\lfloor\frac n2\right\rfloor\right).
$$

The tensor basis isomorphism is

$$
\mathbb Z^6\xrightarrow{\sim}\mathbb Z^2\otimes\mathbb Z^3,
\qquad
e_n\longmapsto
e_{n\bmod2}\otimes e_{\lfloor n/2\rfloor}.
$$

The addresses agree because

$$
n
=
(n\bmod2)+2\left\lfloor\frac n2\right\rfloor.
$$

This is the concrete reason dimension factorization can encode a nested mode.

### Example 9.6: concatenation

Let

$$
A=(2,3):(1,2),
\qquad
B=4:6.
$$

The nested concatenation is

$$
(A,B)=((2,3),4):((1,2),6),
$$

while the flat concatenation is

$$
A\star B=(2,3,4):(1,2,6).
$$

They have different profiles but the same flattening. At the module level,

$$
\mathsf V(A,B)
\cong
\mathsf V(A)\otimes\mathsf V(B)
\cong
(\mathbb Z^2\otimes\mathbb Z^3)\otimes\mathbb Z^4.
$$

At the address level,

$$
\widetilde\ell_{(A,B)}
=
\widetilde\ell_A\boxplus\widetilde\ell_B.
$$

Thus

$$
\ell((i,j),k)
=
\ell_A(i,j)+\ell_B(k)
=
i+2j+6k.
$$

### Example 9.7: permutation

Let

$$
S=(8,16,32,64)
$$

and

$$
\sigma=(0\;1)(2\;3).
$$

Then

$$
S^\sigma=(16,8,64,32).
$$

For a stride

$$
D=(d_0,d_1,d_2,d_3)
$$

and coordinate

$$
c=(c_0,c_1,c_2,c_3),
$$

we have

$$
D^\sigma=(d_1,d_0,d_3,d_2),
\qquad
c^\sigma=(c_1,c_0,c_3,c_2),
$$

and

$$
\ell_{S^\sigma,D^\sigma}(c^\sigma)
=
c_1d_1+c_0d_0+c_3d_3+c_2d_2
=
\ell_{S,D}(c).
$$

On coordinate modules, this is a permutation matrix induced by the symmetric-monoidal braiding.

### Example 9.8: broadcast and aliasing

Take

$$
S=(2,(2,2)),
\qquad
D=(64,(2,0)).
$$

Then

$$
\ell(a,(b,c))=64a+2b.
$$

The last coordinate has zero stride, so

$$
\ell(a,(b,0))=\ell(a,(b,1)).
$$

The linear extension still exists:

$$
\widetilde\ell:
\mathbb Z^2\otimes(\mathbb Z^2\otimes\mathbb Z^2)
\longrightarrow
\mathbb Z.
$$

It is not injective. Tensor realization therefore does not assume compactness, injectivity, or non-aliasing.

---

## 10. Why the tensor formulation helps

### 10.1 It makes size structural

CuTe's multiplicative size law becomes the standard rank law

$$
\operatorname{rank}_R(V\otimes_RW)
=
\operatorname{rank}_R(V)\operatorname{rank}_R(W)
$$

for finite free modules. A mode split $pq\rightsquigarrow(p,q)$ becomes

$$
R^{pq}\xrightarrow{\sim}R^p\otimes_RR^q.
$$

### 10.2 It turns nesting into typed factorization

The expressions

$$
(R^M\otimes R^K)\otimes R^N
$$

and

$$
R^M\otimes(R^K\otimes R^N)
$$

carry different profiles and are connected by a named arrow $\alpha$. This is useful when the parenthesization denotes hardware or algorithmic ownership: thread group, tile, instruction fragment, lane, register, and so on.

### 10.3 It turns permutations into actual linear operators

A mode permutation is no longer informal shuffling. It is an invertible linear map

$$
\beta_\sigma:\mathsf V_R(S)\xrightarrow{\sim}\mathsf V_R(S^\sigma)
$$

whose commutative square with the layout functional certifies address preservation.

### 10.4 It gives every layout a canonical matrix form

Once bases are chosen,

$$
\widetilde\ell_{S,D}:\mathsf V_R(S)\to M
$$

is a matrix. For $M=R$, it is a row vector indexed by complete coordinates. For $M=R^q$, it is a $q\times|S|$ matrix.

Cris's smaller linear form is the matrix of

$$
d_D:\mathsf Q_R(S)\to M.
$$

For $M=R^q$, this is a $q\times\operatorname{len}(S)$ stride matrix, and

$$
\widetilde\ell_{S,D}=d_D\circ\widetilde j_S.
$$

Thus the size-$|S|$ state matrix and the length-$\operatorname{len}(S)$ stride matrix are related, but they are not the same object.

The Colfax lift

$$
R[|\!-\!|]:\mathbf{Nest}\to\mathbf{FMod}_R
$$

instead gives basis-transition matrices that preserve categorical composition. Keeping both matrices prevents the common mistake of conflating coordinate transport with numeric address evaluation.

### 10.5 It exposes proof obligations as diagrams

To claim that a transformation preserves a layout, one proves a square

$$
\begin{array}{ccc}
\mathsf V_R(S)&\xrightarrow{T}&\mathsf V_R(S')\\
\downarrow\widetilde\ell&&\downarrow\widetilde\ell'\\
M&=&M
\end{array}
$$

commutes.

To claim that two transformation paths agree, one proves the corresponding polygon commutes. Associator, unitor, and symmetry-only diagrams commute by coherence; CuTe-specific transformations still require their own arithmetic or categorical proofs.

### 10.6 It generalizes the codomain without changing the shape theory

The same coordinate module can map into:

$$
\mathbb Z,
\qquad
\mathbb Z^q,
\qquad
\mathbb F_2^q,
\qquad
R^q,
$$

depending on whether the layout produces addresses, coordinates, bit patterns, or another module-valued offset.

### 10.7 It gives a construction workflow for new layouts

To design a layout:

1. Choose a profile $P$ reflecting the hardware or algorithmic hierarchy.
2. Label its leaves by positive extents to obtain $S$.
3. Form the coordinate module

   $$
   \mathsf V_R(S)=R[\mathsf C(S)].
   $$

4. Use the nodewise maps $\chi$ to expose the desired tensor factors.
5. Choose a codomain module $M$ and a congruent stride $D$.
6. Build $\widetilde\ell_{S,D}$ from the leaf maps $\lambda_{p,d}$ and augmentations.
7. Apply associators, symmetries, or refinement isomorphisms to derive alternate views.
8. Verify each proposed rewrite by a commuting diagram.
9. Return to the executable layout through

   $$
   [|S|]\xrightarrow{\operatorname{idx2crd}_S}
   \mathsf C(S)\xrightarrow{\eta_S}
   U\mathsf V_R(S)\xrightarrow{U\widetilde\ell_{S,D}}U(M).
   $$

This workflow starts at the vector/module layer without discarding the tuple and profile data required to interpret the factors.

---

## 11. What this formulation does not claim

**Remark 11.1.** It does not identify a tuple with a permutation. Permutations act on tuple positions.

**Remark 11.2.** It does not identify Cartesian product with concatenation or tensor product. The exact chain is

$$
\text{shape concatenation}
\Longrightarrow
\text{Cartesian product of coordinate sets}
\xRightarrow{R[-]}
\text{tensor product of free modules}.
$$

**Remark 11.3.** It does not claim that addition of module elements is disjoint union. Disjoint union linearizes to direct sum:

$$
R[A\bigsqcup B]\cong R[A]\oplus R[B].
$$

Cartesian product linearizes to tensor product:

$$
R[A\times B]\cong R[A]\otimes_RR[B].
$$

**Remark 11.4.** It does not claim that an ordinary associative tensor algebra preserves nested CuTe profiles. It preserves flat concatenation. Profile syntax or an operadic grading is needed for nesting.

**Remark 11.5.** It does not prove that complement, logical division, logical product, or every case of CuTe composition is a tensor operation. The Colfax paper supplies tractability and admissibility conditions for those operations. The present construction supplies a module-valued target in which their realized set maps can be studied.

**Remark 11.6.** It does not claim novelty for the standard algebra. The potentially useful contribution is the particular factorization and bookkeeping:

$$
\text{CuTe profile}
\;+\;
\text{free-module strong monoidality}
\;+\;
\text{layout counit}
\;+\;
\text{Colfax realization}.
$$

---

## 12. Final compressed formulation

Let $\mathsf{HShape}$ be the set of hierarchical positive-integer shapes, regarded as a discrete category. Fix a nonzero commutative unital ring $R$.

Define

$$
\mathsf C:\mathsf{HShape}\longrightarrow\mathbf{FinSet}
$$

recursively by

$$
\mathsf C(p)=[p],
\qquad
\mathsf C(S_0,\ldots,S_{r-1})
=
\prod_j\mathsf C(S_j).
$$

Define

$$
\mathsf V_R:=R[-]\circ\mathsf C.
$$

Then

$$
\mathsf V_R(p)=R^{p}
$$

and

$$
\boxed{
\mathsf V_R(S_0,\ldots,S_{r-1})
\xrightarrow{\sim}
\bigotimes_j\mathsf V_R(S_j).
}
$$

The profile records the tensor-expression tree, while associators, unitors, and symmetries compare different trees.

For an $R$-module $M$ and a stride $D$ congruent to $S$, define

$$
\ell_{S,D}(c)
=
\sum_\lambda[c_\lambda]_Rd_\lambda.
$$

Its unique linear extension is

$$
\boxed{
\widetilde\ell_{S,D}:R[\mathsf C(S)]\longrightarrow M.
}
$$

The executable integral layout is

$$
\boxed{
L_{S,D}
=
U(\widetilde\ell_{S,D})
\circ\eta_S
\circ\operatorname{idx2crd}_S.
}
$$

Flat concatenation yields tensor product of coordinate modules; nested concatenation retains an additional profile node; permutation yields symmetry; reparenthesization yields associators; the Colfax realization functor linearizes as

$$
\boxed{
\mathbf{Nest}
\xrightarrow{|\!-\!|}
\mathbf{FinSet}
\xrightarrow{R[-]}
\mathbf{FMod}_R.
}
$$

That is the airtight core.

---

## References

1. Cris Cecka, [*CuTe Layout Representation and Algebra*](https://arxiv.org/abs/2603.02298), 2026.
2. Jack Carlisle, Jay Shah, Reuben Stern, and Paul VanKoughnett, [*Categorical Foundations for CuTe Layouts*](https://arxiv.org/abs/2601.05972), 2026.
3. Aaron Mazel-Gee and Reuben Stern, *A universal characterization of noncommutative motives and secondary algebraic K-theory*, 2021. This paper is used here as an expository style model--explicit conventions, constructions, observations, diagrams, and theorem boundaries--not as a mathematical source for CuTe.