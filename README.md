# newsrec
News Recommendation Model using GNN.

### Datasets
Raw and preprocessed datasets can be accessed by this link from Google Drive: [datasets](https://drive.google.com/drive/folders/19_hl4deYR4hsySeCoti3a45AS7-GTiV9?usp=sharing).

### Run Model

#### On Colab
Model can be run from Google Colab using the following notebook: [colab.ipynb](https://drive.google.com/file/d/1ExS8Zohr1-SI-yT4nHiFslZ0Gaw21H6S/view?usp=sharing).
You need to sign in with Google account to use Colab.

If you don't want to download datasets, they can be can be mounted directly from the Google Drive within the Colab notebook. Add [datasets](https://drive.google.com/drive/folders/19_hl4deYR4hsySeCoti3a45AS7-GTiV9?usp=sharing) from the 'Shared with me' folder into your Drive by clicking 'Add shortcut to Drive' to access it from the notebook.
Steps on how to run the model is well documented in the notebook along with the code to setup necessary libraries.

#### On Local
Alternatively, `main.ipynb` can be run by downloading the datasets and installing libraries from `requirements.txt`:
```sh
pip3 install -r requirements.txt
```

### References
1. [Pairwise Learning for Neural Link Prediction for OGB](https://github.com/zhitao-wang/PLNLP)

2. [ogbl-ppa](https://github.com/snap-stanford/ogb/tree/bd1cfa20f909f3e0cccf807eb8605961cf3ce49b/examples/linkproppred/ppa)

3. [Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)

4. [Link Prediction on Heterogeneous Graphs with PyG](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70)
