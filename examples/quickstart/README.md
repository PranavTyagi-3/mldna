# mldna quickstart: producer + consumer in 5 minutes

Train a scikit-learn churn model, publish it with full lineage, and fetch it
into a separate "application" - all on your machine. No cloud account, no
server: the git remote, the DVC remote and the model store are local folders.

```
mldna-quickstart/
  remotes/producer.git   git remote the producer pushes code to
  remotes/dvc-store/     DVC remote the producer pushes data to
  model-store/           MLflow store: runs + registry, shared with the consumer
  producer/              where the model is trained and published
  consumer/              the app that pins and fetches a published version
```

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Build the playground

From this folder (`examples/quickstart`). It is created outside the repo, in
`~/mldna-quickstart` (pass another folder as an argument if you like):

```bash
./setup.sh                    # Windows PowerShell: .\setup.ps1
```

## 3. Producer: save code + data, train, publish

```bash
cd ~/mldna-quickstart/producer
mldna stage                   # dvc add the dataset, stage everything, list it
mldna push -m "first version" # shows the file list, asks, commits, pushes code + data
python train.py               # trains inside `with mldna.track_run():`
mldna runs                    # the run, its commit and its accuracy
mldna publish models/model.pkl
```

`publish` matches `models/model.pkl` to the run that saved it by SHA-256, checks
the commit and the data are on their remotes, shows a summary, and registers
`quickstart-churn` v1. It prints the model id - copy it for the next step.

## 4. Consumer: pin, fetch, run

```bash
cd ~/mldna-quickstart/consumer
```

Replace `models.lock.yaml` with (use the id `publish` printed):

```yaml
churn:
  model: quickstart-churn
  id: <model id from publish>
  version: 1
  format: pkl
  dest: ./models/churn
```

```bash
mldna fetch                   # checksum-verified, atomic; re-running skips
python app.py                 # Churn risk for a 24-month customer ...: 22%
```

## 5. Try to break it

- **Train on uncommitted code.** Edit `train.py`, don't commit, run
  `python train.py`: the run is refused, because its recorded commit would not
  be the code that ran.
- **Publish a file no run saved.** In the producer, make a copy with one extra
  byte and publish it - refused, because no run saved those exact bytes (an
  identical copy *would* publish: lineage follows the content, not the name):

  ```bash
  python -c "open('models/edited.pkl','wb').write(open('models/model.pkl','rb').read() + b'x')"
  mldna publish models/edited.pkl
  ```

- **Recall the model.** In the consumer, tag version 1 `recalled`, then fetch -
  it fails, even though the model is already on disk:

  ```bash
  python -c "from mldna.trackinguri import resolve_and_set_tracking_uri; resolve_and_set_tracking_uri('.'); import mlflow; mlflow.MlflowClient().set_model_version_tag('quickstart-churn', '1', 'recalled', 'true')"
  mldna fetch
  ```
- **See what changed.** Change the data or code, `mldna stage`, `mldna push`,
  train again, then `pip install "mldna-lineage[ui]"` and `mldna ui` in the
  producer.

Delete `~/mldna-quickstart` when you're done; nothing else was touched.
