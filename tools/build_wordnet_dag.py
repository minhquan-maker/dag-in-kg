"""Build the WordNet noun is-a DAG from the raw WordNet 3.0 `data.noun` file.

Edges are (child, parent) over synset offsets. Hypernym ('@') and instance
hypernym ('@i') pointers are both used, as in Poincare Embeddings (Nickel &
Kiela 2017) and Hyperbolic Entailment Cones (Ganea et al. 2018).
"""
import argparse, os, zipfile, urllib.request

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"


def load_data_noun(cache):
    os.makedirs(cache, exist_ok=True)
    z = os.path.join(cache, "wordnet.zip")
    if not os.path.exists(z):
        urllib.request.urlretrieve(URL, z)
    with zipfile.ZipFile(z) as f:
        return f.read("wordnet/data.noun").decode("utf8").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/raw")
    ap.add_argument("--out", default="data/wordnet_nouns_edges.tsv")
    a = ap.parse_args()
    names, edges = {}, []
    for line in load_data_noun(a.cache):
        if line.startswith("  "):  # license header
            continue
        t = line.split("|")[0].split()
        off, wcnt = t[0], int(t[3], 16)
        names[off] = t[4].lower()  # first lemma
        p = 4 + 2 * wcnt
        for i in range(int(t[p])):
            sym, tgt, pos = t[p + 1 + 4 * i: p + 4 + 4 * i]
            if sym in ("@", "@i") and pos == "n":
                edges.append((off, tgt))
    with open(a.out, "w") as f:
        for c, p in edges:
            f.write(f"{c}\t{p}\t{names[c]}\t{names[p]}\n")
    print(f"{len(names)} synsets, {len(edges)} is-a edges -> {a.out}")


if __name__ == "__main__":
    main()
