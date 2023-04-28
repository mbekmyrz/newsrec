# newsrec
News Recommendation Model using GNN.

### Datasets
Raw and preprocessed datasets can be accessed by this link from Google Drive: [datasets](https://drive.google.com/drive/folders/19_hl4deYR4hsySeCoti3a45AS7-GTiV9?usp=sharing).

### Run Model

#### On Colab:
You can use `colab.ipynb` to run model on Google Colab or directly go to following link: [colab.ipynb](https://drive.google.com/file/d/1ExS8Zohr1-SI-yT4nHiFslZ0Gaw21H6S/view?usp=sharing).
You need to sign in with Google account to use Colab.

If you don't want to download datasets, they can be can be mounted directly from the Google Drive within the Colab notebook. Add [datasets](https://drive.google.com/drive/folders/19_hl4deYR4hsySeCoti3a45AS7-GTiV9?usp=sharing) from the 'Shared with me' folder into your Drive by clicking 'Add shortcut to Drive' to access it from the notebook.
Steps on how to run the model is well documented in the notebook along with the code to setup necessary libraries.

#### On Local Machine:
Alternatively, `main.ipynb` can be run from local Jupyter Notebook by downloading the datasets and installing libraries from `requirements.txt` and [PyG](https://github.com/pyg-team/pytorch_geometric) from its source:
```sh
pip3 install -r requirements.txt
pip3 install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

With installed libraries, you can also run `main.py` from command line. For more options see `--help`:
```sh
python3 main.py --data cit_pt --plm ptbert --epochs 50 --eval_steps 10
```

### References
[Pairwise Learning for Neural Link Prediction for OGB](https://github.com/zhitao-wang/PLNLP)

[ogbl-ppa](https://github.com/snap-stanford/ogb/tree/bd1cfa20f909f3e0cccf807eb8605961cf3ce49b/examples/linkproppred/ppa)

[Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)

[Link Prediction on Heterogeneous Graphs with PyG](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70)
