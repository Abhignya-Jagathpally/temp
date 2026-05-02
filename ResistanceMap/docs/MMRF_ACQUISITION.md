# MMRF CoMMpass Acquisition — Step-by-Step

The MMRF CoMMpass study (Multiple Myeloma Research Foundation Compass) is a
multi-center longitudinal study of newly-diagnosed MM patients with paired
RNA-seq + WGS + clinical follow-up. It is **controlled-access** — you can't
`curl` it. To use it in this pipeline you need three things:

1. **dbGaP authorization** (one-time, weeks-long IRB process)
2. **A current GDC user-token** (5-minute refresh, expires every 30 days)
3. **A manifest** describing which files to download (5-minute generate)

Then you run `gdc-client download` and the data lands under
`data/raw/mmrf_commpass/` where `configs/default.yaml` already expects it.

The disk has **168 GB free** as of this run; budget the full MMRF release at
**~3-5 TB**. If you only need RNA-seq + clinical, the subset is ~400 GB.

---

## 1. dbGaP authorization

The MMRF data is in dbGaP study **phs000748** ("Relating Clinical Outcomes in
Multiple Myeloma to Personal Assessment of Genetic Profile").

### Prerequisites
- An eRA Commons account tied to your UNT institutional credentials.
  https://www.era.nih.gov/register-accounts/individual-registration-process.htm
- Your PI's name + their eRA Commons ID (you'll list them as the data
  custodian).
- An IRB-approved protocol from UNT covering use of dbGaP-controlled human
  genomic data. If your group already has one for adjacent MM work, attach
  that.

### Submission
1. Sign in to dbGaP: https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/login.cgi
2. **Submit a Data Access Request (DAR)** for phs000748.
3. Fill the Research Use Statement (1-2 paragraphs explaining what you'll
   model — drug response prediction, resistance landscape, etc.).
4. Have your PI co-sign as Authorized User and your Signing Official sign as
   the Institutional SO. UNT's SO contact is in the eRA Commons institution
   profile.
5. Submit. NIH usually approves within **2-6 weeks**.

Track status via the "My Requests" tab in dbGaP.

---

## 2. GDC user-token

Once your DAR is approved, you can mint a GDC token any time. Tokens are
valid for **30 days** and you'll re-mint freely.

1. Log in to the GDC portal with your eRA Commons credentials:
   https://portal.gdc.cancer.gov/
2. Top-right user menu → **Download Token**. A file named
   `gdc-user-token.<timestamp>.txt` lands in your Downloads folder.
3. Move it to this machine and lock it down:
   ```bash
   mkdir -p ~/.gdc
   mv ~/Downloads/gdc-user-token.*.txt ~/.gdc/token.txt
   chmod 600 ~/.gdc/token.txt
   ```
   Do **not** check this file into git. The repo's `.gitignore` already
   ignores `data/raw/` but tokens deserve their own dir under `~/.gdc`.

If you want me (Claude) to help with the token step, I can guide you but I
**cannot** mint it for you — only you can authenticate to NIH.

---

## 3. Manifest

A manifest is a TSV listing the file UUIDs you want. The two ways to build
one:

### Option A: GDC Portal (recommended, point-and-click)
1. Go to https://portal.gdc.cancer.gov/projects/MMRF-COMMPASS
2. **Repository** tab → filter:
   - Project: `MMRF-COMMPASS`
   - Data Category: pick what you need (RNA-Seq, WGS, etc.)
   - Experimental Strategy: e.g. `RNA-Seq` for transcriptome
   - Data Format: `BAM` for raw, `TSV` for normalized counts
3. **Add All Files to Cart** → **Cart** → **Download Manifest**.
4. Save as `~/.gdc/mmrf_manifest.txt`.

### Option B: gdc-client manifest CLI
```bash
gdc-client manifest --project MMRF-COMMPASS --data-category Transcriptome\ Profiling \
    -t ~/.gdc/token.txt -o ~/.gdc/mmrf_manifest.txt
```

Pick A unless you know exactly which UUIDs you want — the portal is more
reliable for getting a complete clinical+omic bundle.

---

## 4. Install gdc-client (this machine)

`gdc-client` is **not currently installed on this host** (`command -v
gdc-client` returns empty). One-time install:

```bash
# Latest binary release (no sudo, drops into ~/.local/bin)
mkdir -p ~/.local/bin
cd /tmp
curl -L -o gdc-client.zip \
    https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64-py3.8-ubuntu-20.04.zip
unzip gdc-client.zip
mv gdc-client ~/.local/bin/
chmod +x ~/.local/bin/gdc-client
~/.local/bin/gdc-client --version   # should print 1.6.1
```

Add `~/.local/bin` to PATH in your `~/.bashrc` if it isn't already.

---

## 5. Download

```bash
mkdir -p data/raw/mmrf_commpass
gdc-client download \
    -m ~/.gdc/mmrf_manifest.txt \
    -t ~/.gdc/token.txt \
    -d data/raw/mmrf_commpass/ \
    --n-processes 4 \
    --retry-amount 5
```

- `--n-processes 4` is conservative; bump to 8 if your bandwidth allows.
- `gdc-client` is **resumable** — kill it and restart and it picks up where
  it left off.
- For the full RNA-seq subset budget **~6-12 hours** depending on bandwidth.

Files arrive nested under `data/raw/mmrf_commpass/<uuid>/<filename>` —
that's fine, the loader walks the tree.

---

## 6. What the pipeline expects

`configs/default.yaml` (already wired) reads:

```yaml
data:
  mmrf_commpass_dir: data/raw/mmrf_commpass/
```

The downstream loader at `resistancemap/data/clinical_labels.py` expects:
- One subdir per patient: `MMRF_<patient-id>_BM_*` (gdc-client lays it out
  this way automatically).
- For each patient: `*.rsem.genes.results` (RNA-seq) and the matching
  clinical TSV (you'll get this in the same manifest if you select the
  Clinical category).

Once the download finishes, **re-run the agentic pipeline from a clean
checkpoints/ dir** so the data-prep stage rebuilds `data_ready.pt` against
the much larger MMRF cohort:

```bash
mv checkpoints checkpoints.bak.pre_mmrf_$(date +%Y%m%dT%H%M%S)
mkdir checkpoints
python main.py --config configs/default.yaml
```

After that, `scripts/run_baselines_real.py` will produce a fresh comparison
table on the MMRF-augmented dataset.

---

## 7. What I (Claude) can help with

| Step | Who does it | Notes |
|---|---|---|
| 1. dbGaP DAR | **You + PI + UNT SO** | I can draft your Research Use Statement if you paste your protocol abstract |
| 2. GDC token | **You** (NIH login) | I can verify the token file format once you put it at `~/.gdc/token.txt` |
| 3. Manifest | You or me | If you paste the portal-generated manifest, I can sanity-check it (file counts, project tag) |
| 4. Install gdc-client | I can do this | Just say the word and I'll run the install commands above |
| 5. Run download | Either | Long-running; I can kick it off in the background and notify you when done |
| 6. Re-run pipeline | I'll do this | Same agentic DAG, will pick up the bigger dataset automatically |

When you're ready, paste me:
- The path of your token file (so I can `chmod 600` and verify it parses)
- The path of your manifest (so I can show file counts before download)

…and I'll run steps 4-6 from there.
