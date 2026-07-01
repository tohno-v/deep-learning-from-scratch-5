# Deep Learning 学習ガイド

このガイドは、このリポジトリを使って Deep Learning、生成モデル、関連する数学、Python コーディングを段階的に学ぶための教材です。対象読者は Python の基本文法を少し知っている人です。NumPy、PyTorch、確率、線形代数は、コードを読みながら補います。

このリポジトリは、`step01` から `step10` までの章別スクリプトと、同じ内容を Jupyter Notebook 形式にした `notebooks` で構成されています。学習では、まず Notebook で全体の流れを見てから、対応する Python ファイルを読むと理解しやすくなります。

## 1. このガイドの目的と使い方

このガイドの目的は次の 4 つです。

- このリポジトリのコードを読む順番を明確にする。
- Deep Learning の基本である「モデル、損失、勾配、更新」を理解する。
- VAE や拡散モデルなどの生成モデルを、数式とコードの両方から理解する。
- 数式の変形を丁寧に追い、Python 実装に落とす感覚を身につける。

学習の進め方は次の順番を推奨します。

1. `README.md` で章構成と依存ライブラリを確認する。
2. `notebooks/01_normal.ipynb` から `notebooks/10_diffusion2.ipynb` までを順に眺める。
3. 各章の `stepXX` フォルダにある Python ファイルを読む。
4. このガイドの「数学の要点」と「コードとの対応」を照らし合わせる。
5. 確認問題を自分の言葉で説明できるか確認する。

このガイドでは、各章を次の固定フォーマットで説明します。

- 学ぶ目的
- 対応するフォルダ・ファイル
- 先に読むべきコード
- 数学の要点
- 数式変形の丁寧な追跡
- コードとの対応
- 確認問題

## 2. リポジトリ全体の構成と読む順番

主要な構成は次のとおりです。

| 場所 | 内容 |
| --- | --- |
| `README.md` | 書籍サポートサイトとしての概要、Notebook へのリンク、依存ライブラリ |
| `requirements.txt` | NumPy、SciPy、Matplotlib、PyTorch、torchvision、tqdm のバージョン |
| `notebooks/` | `step01` から `step10` までの内容を Notebook 化した教材 |
| `step01/` | 正規分布 |
| `step02/` | 最尤推定 |
| `step03/` | 多次元正規分布 |
| `step04/` | 混合ガウスモデル |
| `step05/` | EM アルゴリズム |
| `step06/` | PyTorch とニューラルネットワーク |
| `step07/` | 変分オートエンコーダ |
| `step08/` | 階層 VAE |
| `step09/` | 拡散モデル |
| `step10/` | 条件付き拡散モデルと Classifier-Free Guidance |

読み順は大きく 3 つのブロックに分けられます。

| ブロック | 対象 | 目的 |
| --- | --- | --- |
| 確率と推定の土台 | `step01` から `step05` | 生成モデルで使う確率分布、尤度、混合分布、EM を学ぶ |
| Deep Learning の基本 | `step06` | Tensor、自動微分、勾配降下、ニューラルネット、DataLoader を学ぶ |
| 生成モデル | `step07` から `step10` | VAE、階層 VAE、拡散モデル、条件付き生成を学ぶ |

最初からすべての数式を完全に理解しようとすると重くなります。最初の 1 周目は「この式はコードのどこに対応するか」を探すことを重視してください。2 周目で数式変形を丁寧に追うと、コードの意味がかなり見えます。

## 3. 学習ロードマップ

### Phase 1: 確率分布をコードで描く

対象は `step01` から `step03` です。ここでは、確率分布を数式として見るだけでなく、NumPy 配列として計算し、Matplotlib で可視化します。

この段階での到達目標は次です。

- 正規分布の式を読める。
- データから平均と分散を計算できる。
- 尤度と対数尤度の意味を説明できる。
- 多次元正規分布で共分散行列が何を表すか説明できる。
- `np.linspace`, `np.mean`, `np.cov`, `np.linalg.det`, `np.linalg.inv` の役割を理解する。

### Phase 2: 混合分布と EM を理解する

対象は `step04` と `step05` です。単一の正規分布では表しにくいデータを、複数の正規分布の重ね合わせとして表します。

この段階での到達目標は次です。

- 混合分布 $p(x)=\sum_k \phi_k \mathcal{N}(x|\mu_k,\Sigma_k)$ を読める。
- 潜在変数という考え方を理解する。
- EM アルゴリズムの E-step と M-step の役割を説明できる。
- for ループで数式をそのまま実装する読み方に慣れる。

### Phase 3: PyTorch で学習する

対象は `step06` です。ここから Deep Learning 本体に入ります。

この段階での到達目標は次です。

- `torch.Tensor` と NumPy 配列の違いを理解する。
- `requires_grad=True` と `backward()` の意味を理解する。
- 損失関数を小さくするようにパラメータを更新する流れを説明できる。
- `nn.Module`, `forward`, `optimizer.step()` の役割を理解する。
- `Dataset`, `DataLoader`, `transforms` でデータを扱える。

### Phase 4: 生成モデルを理解する

対象は `step07` から `step10` です。生成モデルでは、入力を分類するだけでなく、データの分布を学び、新しいデータを生成します。

この段階での到達目標は次です。

- VAE の Encoder、潜在変数、Decoder の流れを説明できる。
- 再パラメータ化トリックの必要性を理解する。
- KL 項が「潜在分布を標準正規分布に近づける力」だと説明できる。
- 拡散モデルの forward process と reverse process を区別できる。
- ノイズ予測損失が何を学習しているか説明できる。
- 条件付き生成と Classifier-Free Guidance の直感を説明できる。

## 4. Step 1: 正規分布

### 学ぶ目的

正規分布は、このリポジトリ全体で何度も登場する基本的な確率分布です。VAE の潜在変数、拡散モデルのノイズ、GMM の各成分も正規分布を土台にしています。

### 対応するフォルダ・ファイル

- `step01/norm_dist.py`
- `step01/norm_param.py`
- `step01/sample_avg.py`
- `step01/sample_sum.py`
- `notebooks/01_normal.ipynb`

### 先に読むべきコード

最初は `step01/norm_dist.py` を読みます。短いファイルなので、正規分布の式がそのまま Python 関数になっていることを確認できます。

```python
def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y
```

### 数学の要点

平均 $\mu$、標準偏差 $\sigma$ の正規分布の確率密度関数は次です。

$$
\mathcal{N}(x|\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

この式は 2 つの部分に分けて読むと分かりやすいです。

- $\frac{1}{\sqrt{2\pi}\sigma}$ は全体の面積を 1 にするための係数です。
- $\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ は $x$ が平均 $\mu$ から離れるほど値を小さくします。

### 数式変形の丁寧な追跡

標準正規分布は $\mu=0,\sigma=1$ の場合です。

$$
\mathcal{N}(x|0,1)
=
\frac{1}{\sqrt{2\pi}}
\exp\left(-\frac{x^2}{2}\right)
$$

一般の正規分布では、まず $x$ を平均中心にずらします。

$$
x \rightarrow x-\mu
$$

次に、標準偏差で割ってスケールを合わせます。

$$
z = \frac{x-\mu}{\sigma}
$$

標準正規分布の指数部 $-\frac{z^2}{2}$ に代入すると、

$$
-\frac{z^2}{2}
=
-\frac{1}{2}
\left(\frac{x-\mu}{\sigma}\right)^2
=
-\frac{(x-\mu)^2}{2\sigma^2}
$$

これがコードの `-(x - mu)**2 / (2 * sigma**2)` に対応します。

### コードとの対応

| 数式 | コード |
| --- | --- |
| $x$ の値を並べる | `np.linspace(-5, 5, 100)` |
| $\sqrt{2\pi}$ | `np.sqrt(2 * np.pi)` |
| $\exp(\cdot)$ | `np.exp(...)` |
| $(x-\mu)^2$ | `(x - mu)**2` |
| 曲線を描く | `plt.plot(x, y)` |

`np.linspace(-5, 5, 100)` は、-5 から 5 までを 100 個の点に分けます。数式は連続関数ですが、コンピュータでは有限個の点で値を計算します。

### 確認問題

1. $\mu$ を大きくするとグラフは左右どちらに動くか。
2. $\sigma$ を大きくするとグラフの高さと広がりはどう変わるか。
3. `normal(x, mu=0, sigma=1)` の `x` に NumPy 配列を渡せるのはなぜか。

## 5. Step 2: 最尤推定

### 学ぶ目的

最尤推定は、データから分布のパラメータを推定する方法です。Deep Learning では「損失を小さくする」形で学習しますが、その背景には「データが最も起こりやすいパラメータを探す」という考え方があります。

### 対応するフォルダ・ファイル

- `step02/hist.py`
- `step02/fit.py`
- `step02/generate.py`
- `step02/prob.py`
- `step02/height.txt`
- `notebooks/02_mle.ipynb`

### 先に読むべきコード

最初に `step02/fit.py` を読みます。データを読み込み、平均と標準偏差を計算し、正規分布を当てはめています。

```python
xs = np.loadtxt(path)
mu = np.mean(xs)
sigma = np.std(xs)
```

### 数学の要点

データ $x_1,x_2,\ldots,x_N$ が正規分布 $\mathcal{N}(x|\mu,\sigma^2)$ から独立に得られたとします。このとき尤度は次です。

$$
L(\mu,\sigma)
=
\prod_{n=1}^{N}
\mathcal{N}(x_n|\mu,\sigma^2)
$$

尤度は「そのパラメータなら、手元のデータがどれくらい起こりやすいか」を表します。最尤推定では、この $L(\mu,\sigma)$ を最大にする $\mu,\sigma$ を探します。

### 数式変形の丁寧な追跡

積の形は扱いにくいため、対数を取ります。対数は単調増加関数なので、$L$ を最大にするパラメータと $\log L$ を最大にするパラメータは同じです。

$$
\log L(\mu,\sigma)
=
\log
\prod_{n=1}^{N}
\mathcal{N}(x_n|\mu,\sigma^2)
$$

積の対数は和になります。

$$
\log L(\mu,\sigma)
=
\sum_{n=1}^{N}
\log \mathcal{N}(x_n|\mu,\sigma^2)
$$

正規分布の式を代入します。

$$
\log \mathcal{N}(x_n|\mu,\sigma^2)
=
\log
\left[
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{(x_n-\mu)^2}{2\sigma^2}
\right)
\right]
$$

$\log(ab)=\log a+\log b$ と $\log(\exp a)=a$ を使います。

$$
=
-\log(\sqrt{2\pi}\sigma)
-
\frac{(x_n-\mu)^2}{2\sigma^2}
$$

したがって、

$$
\log L(\mu,\sigma)
=
\sum_{n=1}^{N}
\left[
-\log(\sqrt{2\pi}\sigma)
-
\frac{(x_n-\mu)^2}{2\sigma^2}
\right]
$$

$\mu$ に関係する部分だけ見ると、

$$
-
\sum_{n=1}^{N}
\frac{(x_n-\mu)^2}{2\sigma^2}
$$

です。これを最大化することは、次を最小化することと同じです。

$$
\sum_{n=1}^{N}(x_n-\mu)^2
$$

この値を $\mu$ で微分します。

$$
\frac{\partial}{\partial \mu}
\sum_{n=1}^{N}(x_n-\mu)^2
=
\sum_{n=1}^{N}2(x_n-\mu)(-1)
$$

最小値では微分が 0 です。

$$
-2\sum_{n=1}^{N}(x_n-\mu)=0
$$

両辺を -2 で割ります。

$$
\sum_{n=1}^{N}(x_n-\mu)=0
$$

和を分けます。

$$
\sum_{n=1}^{N}x_n - \sum_{n=1}^{N}\mu = 0
$$

$\mu$ は $n$ に依存しないので、$\sum_{n=1}^{N}\mu=N\mu$ です。

$$
\sum_{n=1}^{N}x_n - N\mu = 0
$$

したがって、

$$
\mu = \frac{1}{N}\sum_{n=1}^{N}x_n
$$

これはコードの `np.mean(xs)` に対応します。

分散についても同様に、最尤推定では次になります。

$$
\sigma^2
=
\frac{1}{N}
\sum_{n=1}^{N}
(x_n-\mu)^2
$$

標準偏差はその平方根です。

$$
\sigma
=
\sqrt{
\frac{1}{N}
\sum_{n=1}^{N}
(x_n-\mu)^2
}
$$

これはコードの `np.std(xs)` に対応します。

### コードとの対応

| 数式・処理 | コード |
| --- | --- |
| データ $x_1,\ldots,x_N$ | `xs = np.loadtxt(path)` |
| 平均 $\mu$ | `np.mean(xs)` |
| 標準偏差 $\sigma$ | `np.std(xs)` |
| 正規分布の当てはめ | `y = normal(x, mu, sigma)` |
| ヒストグラム | `plt.hist(xs, bins='auto', density=True)` |

`density=True` は、ヒストグラムの面積が 1 になるように正規化します。正規分布の確率密度曲線と重ねるために必要です。

### 確認問題

1. 尤度ではなぜ積が出てくるか。
2. 対数尤度にすると、なぜ積が和になるか。
3. `np.mean(xs)` が最尤推定の $\mu$ に対応する理由を説明せよ。

## 6. Step 3: 多次元正規分布

### 学ぶ目的

多次元正規分布は、複数の特徴量を同時に扱うための分布です。身長と体重、画像の潜在表現、ニューラルネットの特徴ベクトルなど、Deep Learning ではベクトルを扱う場面が多くあります。

### 対応するフォルダ・ファイル

- `step03/numpy_basis.py`
- `step03/numpy_matrix.py`
- `step03/plot_3d.py`
- `step03/plot_norm.py`
- `step03/plot_dataset.py`
- `step03/mle.py`
- `step03/height_weight.txt`
- `notebooks/03_multi_normal.ipynb`

### 先に読むべきコード

最初に `step03/numpy_basis.py` と `step03/numpy_matrix.py` で NumPy のベクトル・行列操作を確認します。その後、`step03/mle.py` を読みます。

```python
mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False)
```

### 数学の要点

$d$ 次元ベクトル $x$ の多次元正規分布は次です。

$$
\mathcal{N}(x|\mu,\Sigma)
=
\frac{1}{\sqrt{(2\pi)^d|\Sigma|}}
\exp\left(
-\frac{1}{2}
(x-\mu)^T
\Sigma^{-1}
(x-\mu)
\right)
$$

ここで、各記号の意味は次です。

- $x$: データ点を表すベクトル
- $\mu$: 平均ベクトル
- $\Sigma$: 共分散行列
- $|\Sigma|$: 共分散行列の行列式
- $\Sigma^{-1}$: 共分散行列の逆行列

### 数式変形の丁寧な追跡

1 次元正規分布の指数部は次でした。

$$
-\frac{(x-\mu)^2}{2\sigma^2}
$$

これを少し書き換えます。

$$
-\frac{1}{2}
(x-\mu)
\frac{1}{\sigma^2}
(x-\mu)
$$

1 次元では分散は $\sigma^2$ です。多次元では分散が共分散行列 $\Sigma$ に置き換わります。

$$
\sigma^2 \rightarrow \Sigma
$$

また、割り算に相当するものは逆行列です。

$$
\frac{1}{\sigma^2} \rightarrow \Sigma^{-1}
$$

ベクトル同士の積でスカラーを作るには、転置を使います。

$$
(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

したがって、多次元の指数部は次になります。

$$
-\frac{1}{2}
(x-\mu)^T
\Sigma^{-1}
(x-\mu)
$$

正規化係数も、1 次元の $\sqrt{2\pi}\sigma$ から、多次元の $\sqrt{(2\pi)^d|\Sigma|}$ に変わります。

### コードとの対応

`step03/mle.py` の `multivariate_normal` は、この式をほぼそのまま実装しています。

```python
det = np.linalg.det(cov)
inv = np.linalg.inv(cov)
d = len(x)
z = 1 / np.sqrt((2 * np.pi) ** d * det)
y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
```

| 数式 | コード |
| --- | --- |
| $|\Sigma|$ | `np.linalg.det(cov)` |
| $\Sigma^{-1}$ | `np.linalg.inv(cov)` |
| $d$ | `len(x)` |
| $(x-\mu)^T\Sigma^{-1}(x-\mu)$ | `(x - mu).T @ inv @ (x - mu)` |
| $\exp(\cdot)$ | `np.exp(...)` |

`@` は Python の行列積演算子です。NumPy 配列でベクトルや行列の積を表すときに使います。

### 確認問題

1. 1 次元の分散 $\sigma^2$ は、多次元では何に対応するか。
2. 共分散行列の対角成分と非対角成分は何を表すか。
3. `(x - mu).T @ inv @ (x - mu)` の結果はベクトルかスカラーか。

## 7. Step 4: 混合ガウスモデル

### 学ぶ目的

混合ガウスモデルは、複数の正規分布を重ね合わせて複雑なデータ分布を表すモデルです。生成モデルでは「データは単純な分布 1 つではなく、複数の構造から生まれる」と考えることがよくあります。

### 対応するフォルダ・ファイル

- `step04/old_faithful.py`
- `step04/gmm_sampling.py`
- `step04/gmm.py`
- `step04/old_faithful.txt`
- `notebooks/04_gmm.ipynb`

### 先に読むべきコード

最初に `step04/gmm.py` を読みます。`multivariate_normal` と `gmm` の関係を見ることが重要です。

```python
def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y
```

### 数学の要点

混合ガウスモデルは次の形です。

$$
p(x)
=
\sum_{k=1}^{K}
\phi_k
\mathcal{N}(x|\mu_k,\Sigma_k)
$$

ここで、$K$ は混合する正規分布の数です。$\phi_k$ は混合係数で、次を満たします。

$$
\sum_{k=1}^{K}\phi_k=1
$$

$$
\phi_k \ge 0
$$

直感的には、まずどのクラスタ $k$ から生成するかを $\phi_k$ の確率で選び、そのクラスタの正規分布から $x$ を生成します。

### 数式変形の丁寧な追跡

潜在変数 $z$ を導入します。$z=k$ は「データ $x$ が $k$ 番目の正規分布から生成された」という意味です。

まず、$z=k$ が選ばれる確率を次のように置きます。

$$
p(z=k)=\phi_k
$$

次に、$z=k$ が分かっているときの $x$ の分布を次のように置きます。

$$
p(x|z=k)=\mathcal{N}(x|\mu_k,\Sigma_k)
$$

では、$z$ が分からないときの $x$ の分布はどうなるでしょうか。全確率の公式を使います。

$$
p(x)
=
\sum_{k=1}^{K}
p(x,z=k)
$$

同時確率を条件付き確率で分解します。

$$
p(x,z=k)
=
p(z=k)p(x|z=k)
$$

これを代入します。

$$
p(x)
=
\sum_{k=1}^{K}
p(z=k)p(x|z=k)
$$

$p(z=k)=\phi_k$ と $p(x|z=k)=\mathcal{N}(x|\mu_k,\Sigma_k)$ を代入すると、

$$
p(x)
=
\sum_{k=1}^{K}
\phi_k
\mathcal{N}(x|\mu_k,\Sigma_k)
$$

となります。

### コードとの対応

| 数式 | コード |
| --- | --- |
| $K$ | `K = len(phis)` |
| $\phi_k$ | `phis[k]` |
| $\mu_k$ | `mus[k]` |
| $\Sigma_k$ | `covs[k]` |
| $\mathcal{N}(x|\mu_k,\Sigma_k)$ | `multivariate_normal(x, mu, cov)` |
| $\sum_k$ | `for k in range(K)` |

`y += ...` は、数式の $\sum$ を for ループで実装しています。数式で $\sum$ が出てきたら、コードでは for ループかベクトル化された配列計算になることが多いです。

### 確認問題

1. $\phi_k$ の合計が 1 でなければならない理由を説明せよ。
2. GMM が単一の正規分布より表現力を持つ理由を説明せよ。
3. `for k in range(K)` は数式のどの記号に対応するか。

## 8. Step 5: EM アルゴリズム

### 学ぶ目的

EM アルゴリズムは、潜在変数を含むモデルのパラメータを推定する方法です。GMM では、各データがどの正規分布から来たかが分かりません。EM は、その「分からない所属」を確率として推定しながら、分布のパラメータを更新します。

### 対応するフォルダ・ファイル

- `step05/em.py`
- `step05/generate.py`
- `step05/old_faithful.txt`
- `notebooks/05_em.ipynb`

### 先に読むべきコード

最初に `step05/em.py` を読みます。特に E-step と M-step のコメントがある部分を追います。

```python
# E-step
qs[n, k] = phi * multivariate_normal(x, mu, cov)
qs[n] /= gmm(x, phis, mus, covs)
```

```python
# M-step
phis[k] = qs_sum[k] / N
mus[k] = c / qs_sum[k]
covs[k] = c / qs_sum[k]
```

### 数学の要点

EM アルゴリズムでは、各データ $x_n$ がクラスタ $k$ に属する確率を $q_{nk}$ と置きます。

$$
q_{nk}
=
p(z_n=k|x_n)
$$

これはベイズの定理で計算できます。

$$
p(z_n=k|x_n)
=
\frac{
p(z_n=k)p(x_n|z_n=k)
}{
\sum_{j=1}^{K}
p(z_n=j)p(x_n|z_n=j)
}
$$

GMM の記号で書くと、

$$
q_{nk}
=
\frac{
\phi_k
\mathcal{N}(x_n|\mu_k,\Sigma_k)
}{
\sum_{j=1}^{K}
\phi_j
\mathcal{N}(x_n|\mu_j,\Sigma_j)
}
$$

### 数式変形の丁寧な追跡

E-step は、現在のパラメータで $q_{nk}$ を計算します。

まず、分子を見ます。

$$
\phi_k
\mathcal{N}(x_n|\mu_k,\Sigma_k)
$$

これは「クラスタ $k$ が選ばれる確率」と「そのクラスタから $x_n$ が出る確率」の積です。コードでは次です。

```python
qs[n, k] = phi * multivariate_normal(x, mu, cov)
```

ただし、この時点では正規化されていません。すべての $k$ について合計が 1 になるように、GMM の密度で割ります。

$$
q_{nk}
\leftarrow
\frac{q_{nk}}{\sum_j \phi_j\mathcal{N}(x_n|\mu_j,\Sigma_j)}
$$

コードでは次です。

```python
qs[n] /= gmm(x, phis, mus, covs)
```

M-step では、$q_{nk}$ を重みとしてパラメータを更新します。

混合係数は、クラスタ $k$ に属する割合です。

$$
\phi_k
=
\frac{1}{N}
\sum_{n=1}^{N}q_{nk}
$$

平均は、$q_{nk}$ で重みづけした平均です。

$$
\mu_k
=
\frac{
\sum_{n=1}^{N}q_{nk}x_n
}{
\sum_{n=1}^{N}q_{nk}
}
$$

共分散も同じく重みづけします。

$$
\Sigma_k
=
\frac{
\sum_{n=1}^{N}
q_{nk}
(x_n-\mu_k)(x_n-\mu_k)^T
}{
\sum_{n=1}^{N}q_{nk}
}
$$

この E-step と M-step を交互に繰り返すことで、対数尤度が少しずつ改善されます。

### コードとの対応

| 数式・処理 | コード |
| --- | --- |
| $q_{nk}$ | `qs[n, k]` |
| $\sum_n q_{nk}$ | `qs_sum = qs.sum(axis=0)` |
| $\phi_k$ の更新 | `phis[k] = qs_sum[k] / N` |
| $\mu_k$ の更新 | `mus[k] = c / qs_sum[k]` |
| $\Sigma_k$ の更新 | `covs[k] = c / qs_sum[k]` |
| 対数尤度 | `likelihood(xs, phis, mus, covs)` |

`z = z[:, np.newaxis]` は、ベクトルを列ベクトルに変換するための操作です。これにより `z @ z.T` が外積になり、共分散行列を作れます。

### 確認問題

1. E-step では何を推定しているか。
2. M-step では何を更新しているか。
3. $q_{nk}$ が 0.5 のとき、そのデータはクラスタ $k$ にどの程度寄与するか。

## 9. Step 6: PyTorch とニューラルネットワーク

### 学ぶ目的

ここから Deep Learning 本体に入ります。Deep Learning の基本は、モデルの出力と正解の差を損失として計算し、その損失を小さくするようにパラメータを更新することです。

### 対応するフォルダ・ファイル

- `step06/tensor.py`
- `step06/gradient.py`
- `step06/regression.py`
- `step06/neuralnet.py`
- `step06/vision.py`
- `notebooks/06_pytorch.ipynb`

### 先に読むべきコード

おすすめの順番は次です。

1. `step06/tensor.py`
2. `step06/gradient.py`
3. `step06/regression.py`
4. `step06/neuralnet.py`
5. `step06/vision.py`

`step06/neuralnet.py` では、最小構成のニューラルネットが定義されています。

```python
class Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.linear1(x)
        y = F.sigmoid(y)
        y = self.linear2(y)
        return y
```

### 数学の要点

線形層は次の形です。

$$
y = xW + b
$$

ニューラルネットでは、この線形変換に非線形関数を挟みます。

$$
h = \sigma(xW_1+b_1)
$$

$$
\hat{y} = hW_2+b_2
$$

ここで、$\sigma$ はシグモイド関数です。

$$
\sigma(a)
=
\frac{1}{1+\exp(-a)}
$$

学習では、予測 $\hat{y}$ と正解 $y$ の差を損失にします。平均二乗誤差は次です。

$$
L
=
\frac{1}{N}
\sum_{n=1}^{N}
(y_n-\hat{y}_n)^2
$$

### 数式変形の丁寧な追跡

パラメータを $\theta$ とまとめて書くと、学習の目的は損失 $L(\theta)$ を小さくすることです。

勾配降下法では、現在のパラメータから、損失が増える方向の逆向きへ少し動かします。

$$
\theta
\leftarrow
\theta
-
\eta
\frac{\partial L}{\partial \theta}
$$

ここで、$\eta$ は学習率です。

1 変数の簡単な例で見ると、$L(w)=(w-3)^2$ の微分は次です。

$$
\frac{dL}{dw}
=
2(w-3)
$$

$w=0$ のとき、

$$
\frac{dL}{dw}
=
2(0-3)
=
-6
$$

更新式に入れると、

$$
w
\leftarrow
0
-
\eta(-6)
=
6\eta
$$

$w$ は 3 に近づく方向へ動きます。PyTorch はこの微分計算を `backward()` で自動的に行います。

### コードとの対応

| 数式・処理 | コード |
| --- | --- |
| Tensor を作る | `torch.tensor(...)` |
| 勾配を追跡する | `requires_grad=True` |
| 損失 $L$ | `loss = F.mse_loss(y, y_pred)` |
| 勾配計算 | `loss.backward()` |
| 勾配を初期化 | `optimizer.zero_grad()` |
| パラメータ更新 | `optimizer.step()` |
| モデル定義 | `class Model(nn.Module)` |
| 順伝播 | `forward(self, x)` |

`step06/regression.py` では手動で更新しています。

```python
W.data -= lr * W.grad.data
b.data -= lr * b.grad.data
```

`step06/neuralnet.py` では optimizer に任せています。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer.step()
```

この 2 つを比べると、PyTorch の optimizer が「勾配を使ってパラメータを更新する処理」を一般化していることが分かります。

### 確認問題

1. `requires_grad=True` は何のために必要か。
2. `loss.backward()` の後、どこに勾配が保存されるか。
3. `optimizer.zero_grad()` を忘れると何が起きるか。
4. `forward` はいつ呼ばれるか。

## 10. Step 7: 変分オートエンコーダ

### 学ぶ目的

VAE は、画像などのデータを潜在変数 $z$ に変換し、その $z$ から元のデータを再構成する生成モデルです。通常のオートエンコーダと違い、潜在変数を確率分布として扱います。

### 対応するフォルダ・ファイル

- `step07/vae.py`
- `notebooks/07_vae.ipynb`

### 先に読むべきコード

`step07/vae.py` では、`Encoder`, `Decoder`, `reparameterize`, `VAE` の順に読むと分かりやすいです。

```python
mu, sigma = self.encoder(x)
z = reparameterize(mu, sigma)
x_hat = self.decoder(z)
```

### 数学の要点

VAE では、Encoder が入力 $x$ から潜在変数 $z$ の分布を出します。

$$
q_\phi(z|x)
=
\mathcal{N}(z|\mu_\phi(x),\sigma_\phi(x)^2)
$$

Decoder は、潜在変数 $z$ からデータ $x$ を生成します。

$$
p_\theta(x|z)
$$

学習では、大きく 2 つの項を最適化します。

- 再構成誤差: $x$ と $\hat{x}$ を近づける。
- KL 項: $q_\phi(z|x)$ を標準正規分布 $p(z)=\mathcal{N}(0,I)$ に近づける。

コードでは、損失が次のように書かれています。

```python
L1 = F.mse_loss(x_hat, x, reduction='sum')
L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
return (L1 + L2) / batch_size
```

### 数式変形の丁寧な追跡

VAE では、$z$ を正規分布からサンプリングします。

$$
z \sim \mathcal{N}(\mu,\sigma^2)
$$

しかし、単にサンプリングすると、乱数の操作が入るため、Encoder のパラメータへ勾配を流しにくくなります。そこで、標準正規分布からのノイズ $\epsilon$ を使って次のように書き換えます。

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
z = \mu + \sigma \epsilon
$$

この形なら、乱数は $\epsilon$ に分離され、$\mu$ と $\sigma$ は通常の微分可能な演算として扱えます。これが再パラメータ化トリックです。

コードでは次です。

```python
eps = torch.randn_like(sigma)
z = mu + eps * sigma
```

次に KL 項を見ます。1 次元の正規分布 $q(z)=\mathcal{N}(\mu,\sigma^2)$ と標準正規分布 $p(z)=\mathcal{N}(0,1)$ の KL ダイバージェンスは次です。

$$
D_{KL}(q||p)
=
\frac{1}{2}
\left(
\mu^2+\sigma^2-\log\sigma^2-1
\right)
$$

コードの `L2` は次でした。

$$
-
\sum
\left(
1+\log\sigma^2-\mu^2-\sigma^2
\right)
$$

符号を中に入れると、

$$
\sum
\left(
-1-\log\sigma^2+\mu^2+\sigma^2
\right)
$$

並べ替えると、

$$
\sum
\left(
\mu^2+\sigma^2-\log\sigma^2-1
\right)
$$

これは KL 項の 2 倍に相当する形です。定数倍は学習のバランスに影響しますが、ここでは「標準正規分布から離れるほど損失が増える項」として理解します。

### コードとの対応

| 数式・処理 | コード |
| --- | --- |
| $\mu_\phi(x)$ | `self.linear_mu(h)` |
| $\log\sigma^2_\phi(x)$ | `self.linear_logvar(h)` |
| $\sigma$ | `torch.exp(0.5 * logvar)` |
| $\epsilon$ | `torch.randn_like(sigma)` |
| $z=\mu+\sigma\epsilon$ | `z = mu + eps * sigma` |
| Decoder | `self.decoder(z)` |
| 再構成誤差 | `F.mse_loss(x_hat, x, reduction='sum')` |
| KL 項 | `L2 = - torch.sum(...)` |

### 確認問題

1. VAE の Encoder は 1 つのベクトルではなく何を出力しているか。
2. 再パラメータ化トリックはなぜ必要か。
3. KL 項が 0 に近いとき、潜在分布は何に近いか。

## 11. Step 8: 階層 VAE

### 学ぶ目的

階層 VAE は、潜在変数を 1 段だけでなく複数段にしたモデルです。より抽象的な潜在変数 $z_2$ から $z_1$ を生成し、$z_1$ から画像 $x$ を生成します。

### 対応するフォルダ・ファイル

- `step08/hvae.py`
- `notebooks/08_hvae.ipynb`

### 先に読むべきコード

`step08/hvae.py` の `VAE.get_loss` を中心に読みます。

```python
mu1, sigma1 = self.encoder1(x)
z1 = reparameterize(mu1, sigma1)
mu2, sigma2 = self.encoder2(z1)
z2 = reparameterize(mu2, sigma2)

z_hat = self.decoder2(z2)
x_hat = self.decoder1(z1)
```

### 数学の要点

通常の VAE は次の流れです。

$$
x \rightarrow z \rightarrow \hat{x}
$$

階層 VAE では次の流れになります。

$$
x \rightarrow z_1 \rightarrow z_2
$$

生成方向では逆向きに、

$$
z_2 \rightarrow z_1 \rightarrow x
$$

と考えます。

### 数式変形の丁寧な追跡

通常の VAE の損失は、おおまかに次でした。

$$
L
=
\text{reconstruction}
+
\text{KL}(q(z|x)||p(z))
$$

階層 VAE では潜在変数が 2 段になるので、KL に相当する項も複数になります。

コードでは次の 3 つの項が使われています。

```python
L1 = F.mse_loss(x_hat, x, reduction='sum')
L2 = - torch.sum(1 + torch.log(sigma2 ** 2) - mu2 ** 2 - sigma2 ** 2)
L3 = - torch.sum(1 + torch.log(sigma1 ** 2) - (mu1 - z_hat) ** 2 - sigma1 ** 2)
```

`L1` は画像の再構成誤差です。

$$
L_1
=
\|x-\hat{x}\|^2
$$

`L2` は $z_2$ を標準正規分布に近づける項です。

$$
L_2
\approx
D_{KL}
\left(
q(z_2|z_1)
||
\mathcal{N}(0,I)
\right)
$$

`L3` は $z_1$ の分布を、$z_2$ から予測した $z_1$ の分布に近づける項です。コードでは `z_hat = self.decoder2(z2)` が $z_2$ から予測した $z_1$ の平均のように働いています。

$$
L_3
\approx
D_{KL}
\left(
q(z_1|x)
||
p(z_1|z_2)
\right)
$$

`(mu1 - z_hat) ** 2` は、$q(z_1|x)$ の平均 $\mu_1$ と、$p(z_1|z_2)$ 側の平均に相当する `z_hat` の差を表しています。

### コードとの対応

| 概念 | コード |
| --- | --- |
| $q(z_1|x)$ | `self.encoder1(x)` |
| $q(z_2|z_1)$ | `self.encoder2(z1)` |
| $p(z_1|z_2)$ | `self.decoder2(z2)` |
| $p(x|z_1)$ | `self.decoder1(z1)` |
| 画像再構成 | `x_hat = self.decoder1(z1)` |
| 新規生成 | `z2 = torch.randn(sample_size, latent_dim)` |

生成時はまず標準正規分布から $z_2$ をサンプリングします。

```python
z2 = torch.randn(sample_size, latent_dim)
z1_hat = model.decoder2(z2)
z1 = reparameterize(z1_hat, torch.ones_like(z1_hat))
x = model.decoder1(z1)
```

これは、$z_2 \rightarrow z_1 \rightarrow x$ の生成方向に対応します。

### 確認問題

1. 通常の VAE と階層 VAE の違いは何か。
2. `decoder2` は画像を直接生成しているか。
3. 生成時に最初にサンプリングするのは $z_1$ と $z_2$ のどちらか。

## 12. Step 9: 拡散モデル

### 学ぶ目的

拡散モデルは、データに少しずつノイズを加える過程を定義し、その逆過程をニューラルネットで学習する生成モデルです。現在の画像生成モデルの重要な基礎です。

### 対応するフォルダ・ファイル

- `step09/simple_unet.py`
- `step09/gaussian_noise.py`
- `step09/diffusion_model.py`
- `step09/flower.png`
- `notebooks/09_diffusion.ipynb`

### 先に読むべきコード

読む順番は次を推奨します。

1. `step09/gaussian_noise.py`
2. `step09/simple_unet.py`
3. `step09/diffusion_model.py`

`step09/diffusion_model.py` では、`Diffuser.add_noise`, `UNet`, 学習ループ、`Diffuser.sample` を順に読みます。

### 数学の要点

拡散モデルの forward process では、元画像 $x_0$ にノイズを加えて $x_t$ を作ります。

$$
q(x_t|x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

この式は、次の形でサンプリングできます。

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

ここで、

$$
\epsilon \sim \mathcal{N}(0,I)
$$

です。

モデルは、ノイズ入り画像 $x_t$ と時刻 $t$ から、加えたノイズ $\epsilon$ を予測します。

$$
\epsilon_\theta(x_t,t)
\approx
\epsilon
$$

学習損失は次です。

$$
L
=
\mathbb{E}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
$$

### 数式変形の丁寧な追跡

各時刻で少しだけノイズを加える係数を $\beta_t$ とします。

$$
\alpha_t = 1-\beta_t
$$

何ステップも繰り返したときの累積を次で表します。

$$
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s
$$

これにより、$x_0$ から任意の $x_t$ を一気に作れます。

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

この式の直感は次です。

- $\sqrt{\bar{\alpha}_t}x_0$ は元画像の残り具合です。
- $\sqrt{1-\bar{\alpha}_t}\epsilon$ は混ぜるノイズの量です。
- $t$ が大きいほど $\bar{\alpha}_t$ は小さくなり、元画像の情報が減ります。

コードの `add_noise` はこの式を実装しています。

```python
alpha_bar = self.alpha_bars[t_idx]
alpha_bar = alpha_bar.view(N, 1, 1, 1)
noise = torch.randn_like(x_0, device=self.device)
x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
```

学習では、モデルに $x_t$ と $t$ を渡します。

```python
noise_pred = model(x_noisy, t)
loss = F.mse_loss(noise, noise_pred)
```

つまり、モデルは画像そのものではなく、まず「どんなノイズが乗っているか」を当てるように学習します。

### コードとの対応

| 数式・処理 | コード |
| --- | --- |
| $\beta_t$ | `self.betas` |
| $\alpha_t=1-\beta_t$ | `self.alphas = 1 - self.betas` |
| $\bar{\alpha}_t=\prod_s\alpha_s$ | `torch.cumprod(self.alphas, dim=0)` |
| $\epsilon$ | `torch.randn_like(x_0)` |
| $x_t$ | `x_t = ...` |
| $\epsilon_\theta(x_t,t)$ | `model(x_noisy, t)` |
| ノイズ予測損失 | `F.mse_loss(noise, noise_pred)` |
| 逆生成 | `diffuser.sample(model)` |

`UNet` では、時刻 $t$ を positional encoding に変換して畳み込みブロックに渡しています。

```python
v = pos_encoding(timesteps, self.time_embed_dim, x.device)
```

拡散モデルでは、同じ画像でも時刻 $t$ によってノイズの量が違います。そのため、モデルには画像だけでなく時刻情報も必要です。

### 確認問題

1. 拡散モデルの forward process では何をしているか。
2. モデルは $x_0$ を直接予測しているか、それともノイズを予測しているか。
3. 時刻 $t$ をモデルに渡す必要がある理由を説明せよ。

## 13. Step 10: 条件付き生成と Classifier-Free Guidance

### 学ぶ目的

条件付き生成では、「数字の 3 を生成したい」のように、ラベルやテキストなどの条件を指定して生成します。Classifier-Free Guidance は、条件付き生成の強さを調整するための方法です。

### 対応するフォルダ・ファイル

- `step10/conditional.py`
- `step10/classifier_free_guidance.py`
- `notebooks/10_diffusion2.ipynb`

### 先に読むべきコード

最初に `step10/conditional.py` でラベル埋め込みを確認します。その後、`step10/classifier_free_guidance.py` で条件あり予測と条件なし予測の組み合わせを読みます。

```python
eps = model(x, t, labels)
eps_uncond = model(x, t)
eps = eps_uncond + gamma * (eps - eps_uncond)
```

### 数学の要点

条件付き拡散モデルでは、ノイズ予測モデルに条件 $y$ を渡します。

$$
\epsilon_\theta(x_t,t,y)
$$

条件なしの場合は次です。

$$
\epsilon_\theta(x_t,t)
$$

Classifier-Free Guidance では、この 2 つを組み合わせます。

$$
\hat{\epsilon}
=
\epsilon_\theta(x_t,t)
+
\gamma
\left(
\epsilon_\theta(x_t,t,y)
-
\epsilon_\theta(x_t,t)
\right)
$$

$\gamma$ は guidance scale です。大きくすると条件を強く反映しますが、強すぎると画像の自然さが崩れることがあります。

### 数式変形の丁寧な追跡

まず、条件なし予測を $\epsilon_{\text{uncond}}$ と置きます。

$$
\epsilon_{\text{uncond}}
=
\epsilon_\theta(x_t,t)
$$

条件あり予測を $\epsilon_{\text{cond}}$ と置きます。

$$
\epsilon_{\text{cond}}
=
\epsilon_\theta(x_t,t,y)
$$

条件によって変わった方向は、差分で表せます。

$$
\epsilon_{\text{cond}}
-
\epsilon_{\text{uncond}}
$$

この差分を $\gamma$ 倍して、条件なし予測に足します。

$$
\hat{\epsilon}
=
\epsilon_{\text{uncond}}
+
\gamma
(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}})
$$

$\gamma=1$ のときは、

$$
\hat{\epsilon}
=
\epsilon_{\text{uncond}}
+
(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}})
=
\epsilon_{\text{cond}}
$$

つまり通常の条件付き予測と同じです。

$\gamma>1$ のときは、条件付き方向をさらに強調します。

コードでは次です。

```python
eps = model(x, t, labels)
eps_uncond = model(x, t)
eps = eps_uncond + gamma * (eps - eps_uncond)
```

### コードとの対応

| 概念 | コード |
| --- | --- |
| ラベル埋め込み | `self.label_emb = nn.Embedding(num_labels, time_embed_dim)` |
| 時刻埋め込み | `pos_encoding(timesteps, self.time_embed_dim)` |
| 条件を足す | `t += self.label_emb(labels)` |
| 条件あり予測 | `model(x, t, labels)` |
| 条件なし予測 | `model(x, t)` |
| guidance scale | `gamma` |
| CFG の合成 | `eps_uncond + gamma * (eps - eps_uncond)` |

学習中に一定確率で `labels = None` にしている点も重要です。

```python
if np.random.random() < 0.1:
    labels = None
```

これにより、同じモデルが条件あり予測と条件なし予測の両方を学習できます。Classifier-Free Guidance という名前は、別の classifier を使わずに guidance を行うことに由来します。

### 確認問題

1. 条件付き生成でラベルをモデルに渡す理由を説明せよ。
2. $\gamma=1$ のとき、CFG の式は何になるか。
3. 学習中にラベルをあえて消す理由を説明せよ。

## 14. Python コーディング学習

このリポジトリは、Deep Learning だけでなく Python コーディングの教材としても使えます。特に、数式を配列計算や学習ループに落とす読み方を学べます。

### NumPy の読み方

NumPy は、数式を配列として計算するためのライブラリです。

| コード | 意味 | 登場箇所 |
| --- | --- | --- |
| `np.linspace(a, b, n)` | `a` から `b` までを `n` 点に分割 | `step01`, `step02` |
| `np.mean(xs)` | 平均を計算 | `step02` |
| `np.std(xs)` | 標準偏差を計算 | `step02` |
| `np.cov(xs, rowvar=False)` | 共分散行列を計算 | `step03` |
| `np.linalg.det(cov)` | 行列式を計算 | `step03` |
| `np.linalg.inv(cov)` | 逆行列を計算 | `step03` |
| `np.meshgrid(xs, ys)` | 2 次元グリッドを作る | `step03`, `step04` |

数式の $\sum$ は、最初は for ループで読めば十分です。慣れてきたら、NumPy の配列演算でまとめて計算できることを確認します。

### PyTorch の読み方

PyTorch は、NumPy のような配列計算に加えて、自動微分とニューラルネット構築を提供します。

| コード | 意味 | 登場箇所 |
| --- | --- | --- |
| `torch.tensor(...)` | Tensor を作る | `step06/tensor.py` |
| `requires_grad=True` | 勾配を追跡する | `step06/tensor.py` |
| `loss.backward()` | 勾配を計算する | `step06` 以降 |
| `nn.Module` | モデルを定義する基底クラス | `step06` 以降 |
| `forward` | 順伝播の計算 | `step06` 以降 |
| `optimizer.step()` | パラメータ更新 | `step06` 以降 |
| `DataLoader` | ミニバッチを作る | `step06` 以降 |
| `transforms.ToTensor()` | 画像を Tensor に変換 | `step06` 以降 |

### 学習ループの基本形

Deep Learning の学習ループは、多くの場合、次の形になります。

```python
for epoch in range(epochs):
    for x, label in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
```

このリポジトリでは、VAE や拡散モデルでも基本形は同じです。違うのは、`model(x)` の入力や、損失の作り方です。

### 可視化コードの読み方

Matplotlib は、数式や学習結果を目で確認するために使います。

| コード | 用途 |
| --- | --- |
| `plt.plot(x, y)` | 曲線を描く |
| `plt.scatter(x, y)` | 散布図を描く |
| `plt.hist(xs, density=True)` | ヒストグラムを描く |
| `plt.contour(X, Y, Z)` | 等高線を描く |
| `plt.imshow(image, cmap='gray')` | 画像を表示する |
| `plt.show()` | 図を表示する |

可視化は単なる飾りではありません。確率分布や生成結果を確認し、数式やモデルが何をしているかを直感的に理解するための重要な手段です。

## 15. 章ごとの復習チェックリスト

### Step 1: 正規分布

- 正規分布の式をコードで説明できる。
- $\mu$ と $\sigma$ を変えたときのグラフの変化を説明できる。
- `np.linspace` と `plt.plot` の役割を説明できる。

### Step 2: 最尤推定

- 尤度と対数尤度の違いを説明できる。
- 積の対数が和になることを使える。
- 平均の最尤推定が標本平均になる変形を追える。

### Step 3: 多次元正規分布

- 共分散行列の役割を説明できる。
- 行列式と逆行列が多次元正規分布のどこに出てくるか説明できる。
- `@` による行列積を読める。

### Step 4: GMM

- 混合係数 $\phi_k$ の意味を説明できる。
- $\sum_k \phi_k \mathcal{N}(x|\mu_k,\Sigma_k)$ を for ループとして読める。
- 潜在変数 $z$ の意味を説明できる。

### Step 5: EM

- E-step と M-step の違いを説明できる。
- $q_{nk}$ が何を表すか説明できる。
- 重み付き平均として $\mu_k$ の更新式を説明できる。

### Step 6: PyTorch

- Tensor と自動微分の基本を説明できる。
- `backward()` と `optimizer.step()` の役割を説明できる。
- `nn.Module` でモデルを読むことができる。

### Step 7: VAE

- Encoder と Decoder の役割を説明できる。
- 再パラメータ化トリックを式とコードで説明できる。
- 再構成誤差と KL 項の役割を説明できる。

### Step 8: 階層 VAE

- 潜在変数が複数段になる意味を説明できる。
- $z_2 \rightarrow z_1 \rightarrow x$ の生成方向を説明できる。
- `decoder1` と `decoder2` の違いを説明できる。

### Step 9: 拡散モデル

- $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ を説明できる。
- モデルがノイズを予測していることを説明できる。
- 時刻埋め込みが必要な理由を説明できる。

### Step 10: 条件付き生成

- ラベル埋め込みの役割を説明できる。
- Classifier-Free Guidance の式を説明できる。
- `gamma` を大きくしたときの効果を説明できる。

## 16. 演習

### 演習 1: 正規分布のパラメータを変える

`step01/norm_param.py` を読み、$\mu$ と $\sigma$ を変更したときのグラフの変化を説明してください。

確認すること:

- $\mu$ は分布の中心を変える。
- $\sigma$ は分布の広がりを変える。
- $\sigma$ が大きくなるとピークは低くなる。

### 演習 2: 最尤推定を自分で説明する

`step02/fit.py` の次の 2 行が、なぜ最尤推定になるのかを説明してください。

```python
mu = np.mean(xs)
sigma = np.std(xs)
```

説明では、尤度、対数尤度、二乗誤差の最小化を使ってください。

### 演習 3: 多次元正規分布のコードを式に戻す

`step03/mle.py` の次の行を数式に戻してください。

```python
y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
```

確認すること:

- `z` は正規化係数。
- `inv` は共分散行列の逆行列。
- `@` は行列積。

### 演習 4: EM の E-step を手計算する

データ点 1 つ $x$ に対して、クラスタが 2 つあるとします。次の値が与えられたとき、$q_1$ と $q_2$ を計算してください。

$$
\phi_1 \mathcal{N}_1(x)=0.2
$$

$$
\phi_2 \mathcal{N}_2(x)=0.8
$$

答えは次です。

$$
q_1=\frac{0.2}{0.2+0.8}=0.2
$$

$$
q_2=\frac{0.8}{0.2+0.8}=0.8
$$

### 演習 5: PyTorch の学習ループを説明する

`step06/neuralnet.py` の学習ループを、次の言葉を使って説明してください。

- 予測
- 損失
- 勾配
- 初期化
- 更新

### 演習 6: VAE の損失を分解する

`step07/vae.py` の `get_loss` を読み、`L1` と `L2` がそれぞれ何を表すか説明してください。

確認すること:

- `L1` は再構成誤差。
- `L2` は潜在分布を標準正規分布に近づける項。
- `reparameterize` によりサンプリングを含む計算でも勾配が流れる。

### 演習 7: 拡散モデルのノイズ付加を説明する

`step09/diffusion_model.py` の `add_noise` を読み、次の式との対応を説明してください。

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

確認すること:

- `alpha_bar` は時刻ごとに異なる。
- `view(N, 1, 1, 1)` は画像テンソルにブロードキャストするため。
- `noise` が教師信号として使われる。

### 演習 8: Classifier-Free Guidance を説明する

`step10/classifier_free_guidance.py` の次の式を説明してください。

```python
eps = eps_uncond + gamma * (eps - eps_uncond)
```

確認すること:

- `eps_uncond` は条件なし予測。
- `eps` は最初、条件あり予測。
- 差分が条件の方向を表す。
- `gamma` が条件の強さを調整する。

## 17. 学習時の注意点

このリポジトリのコードは、教材として読みやすい単一スクリプト構成です。実務用のパッケージ構成やテスト構成ではありません。そのため、まずは 1 ファイルずつ上から下へ読んで、数式とコードの対応を理解することを優先してください。

また、一部のスクリプトは MNIST データセットをダウンロードします。実行環境によってはネットワーク接続や保存先の確認が必要です。このガイドでは実行環境の構築ではなく、コードを読んで理解することを主目的にしています。

深層学習の学習では、数式、コード、図の 3 つを往復することが重要です。

- 数式だけ読むと、何を計算しているかは分かっても実装が見えにくい。
- コードだけ読むと、なぜその計算をしているかが見えにくい。
- 図だけ見ると、直感は得られても再現できない。

このリポジトリでは、数式をコードにし、コードの結果を図で確認する流れが一貫しています。各章で「式のどの部分が、どの変数や演算に対応しているか」を確認しながら進めてください。
