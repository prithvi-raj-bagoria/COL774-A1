SHELL := /bin/bash
PY    := python3

# ============================================================
# EDIT ME — paths to your Kaggle dataset
# ============================================================
DATA_DIR   := /kaggle/input/datasets/prithviraj15/dataset-col774-a1
PARTD_DIR  := $(DATA_DIR)/cgm_npz_partd/Partd

TRAIN_CSV  := $(DATA_DIR)/e4_hr_train_downsampled.csv
TEST_CSV   := $(DATA_DIR)/e4_hr_test_downsampled.csv
REG        := $(DATA_DIR)/regularization.txt
FOLDS      := $(DATA_DIR)/folds.txt          # EDIT ME if the filename/location differs

WORK       := /kaggle/working/work
SRC        := src
EVAL       := evaluation

SUBJECT_TRAIN := c1s01 c1s02 c1s03 c1s05 c2s01 c2s02 c2s04 c2s05
SUBJECT_TEST  := c1s04 c2s03

.PHONY: all clean \
        parta partb partc \
        prepare-d1 train-d1 fe-d1 eval-d1 d1 \
        prepare-d2 train-d2 fe-d2 eval-d2 d2 \
        prepare-d3 train-d3 fe-d3 eval-d3 d3 \
        partd-all

all: parta partb partc partd-all

# ============================================================
# Part (a) — closed-form linear regression
# ============================================================
parta:
	$(PY) $(SRC)/part_a.py $(TRAIN_CSV) $(TEST_CSV) predictions_a.txt weights_a.txt

# ============================================================
# Part (b) — ridge regression + 5-fold CV
# ============================================================
partb:
	$(PY) $(SRC)/part_b.py $(TRAIN_CSV) $(TEST_CSV) $(FOLDS) $(REG) \
		predictions_b.txt weights_b.txt bestlambda_b.txt crossvalidation_errors_b.txt

# ============================================================
# Part (c) — feature engineering for HR prediction
# ============================================================
partc:
	$(PY) $(SRC)/part_c.py $(TRAIN_CSV) $(TEST_CSV) predictions_c.txt

# ============================================================
# Part (d1) — random within-subject split
# ============================================================
prepare-d1:
	rm -rf $(WORK)/d1
	mkdir -p $(WORK)/d1/labels
	for f in $(PARTD_DIR)/random_test/*.npz; do \
		ln -sf "$$(readlink -f "$$f")" "$(WORK)/d1/labels/$$(basename "$$f")"; \
	done

train-d1: prepare-d1
	time $(PY) $(SRC)/part_d.py train d1 $(PARTD_DIR)/random_train model_d1.pkl

fe-d1:
	$(PY) $(SRC)/part_d.py feature_engineering d1 $(PARTD_DIR)/random_test model_d1.pkl features_d1.npy

eval-d1:
	$(PY) $(EVAL)/eval_d.py --model model_d1.pkl --features features_d1.npy \
		--labels $(WORK)/d1/labels --top-features 5

d1: train-d1 fe-d1 eval-d1

# ============================================================
# Part (d2) — temporal within-subject split (*_a.npz = train, *_b.npz = test)
# ============================================================
prepare-d2:
	rm -rf $(WORK)/d2
	mkdir -p $(WORK)/d2/train $(WORK)/d2/test $(WORK)/d2/labels
	for f in $(PARTD_DIR)/train_set/*_a.npz $(PARTD_DIR)/test_set/*_a.npz; do \
		[ -e "$$f" ] && ln -sf "$$(readlink -f "$$f")" "$(WORK)/d2/train/$$(basename "$$f")"; \
	done
	for f in $(PARTD_DIR)/train_set/*_b.npz $(PARTD_DIR)/test_set/*_b.npz; do \
		[ -e "$$f" ] && ln -sf "$$(readlink -f "$$f")" "$(WORK)/d2/test/$$(basename "$$f")"; \
	done
	for f in $(WORK)/d2/test/*.npz; do \
		ln -sf "$$(readlink -f "$$f")" "$(WORK)/d2/labels/$$(basename "$$f")"; \
	done

train-d2: prepare-d2
	time $(PY) $(SRC)/part_d.py train d2 $(WORK)/d2/train model_d2.pkl

fe-d2:
	$(PY) $(SRC)/part_d.py feature_engineering d2 $(WORK)/d2/test model_d2.pkl features_d2.npy

eval-d2:
	$(PY) $(EVAL)/eval_d.py --model model_d2.pkl --features features_d2.npy \
		--labels $(WORK)/d2/labels --top-features 5

d2: train-d2 fe-d2 eval-d2

# ============================================================
# Part (d3) — cross-subject split (held-out subjects entirely in test)
# ============================================================
prepare-d3:
	rm -rf $(WORK)/d3
	mkdir -p $(WORK)/d3/train $(WORK)/d3/test $(WORK)/d3/labels
	for s in $(SUBJECT_TRAIN); do \
		ln -sf "$$(readlink -f "$(PARTD_DIR)/train_set/$${s}_a.npz")" "$(WORK)/d3/train/$${s}_a.npz"; \
		ln -sf "$$(readlink -f "$(PARTD_DIR)/test_set/$${s}_b.npz")"  "$(WORK)/d3/train/$${s}_b.npz"; \
	done
	for s in $(SUBJECT_TEST); do \
		ln -sf "$$(readlink -f "$(PARTD_DIR)/train_set/$${s}_a.npz")" "$(WORK)/d3/test/$${s}_a.npz"; \
		ln -sf "$$(readlink -f "$(PARTD_DIR)/test_set/$${s}_b.npz")"  "$(WORK)/d3/test/$${s}_b.npz"; \
	done
	for f in $(WORK)/d3/test/*.npz; do \
		ln -sf "$$(readlink -f "$$f")" "$(WORK)/d3/labels/$$(basename "$$f")"; \
	done

train-d3: prepare-d3
	time $(PY) $(SRC)/part_d.py train d3 $(WORK)/d3/train model_d3.pkl

fe-d3:
	$(PY) $(SRC)/part_d.py feature_engineering d3 $(WORK)/d3/test model_d3.pkl features_d3.npy

eval-d3:
	$(PY) $(EVAL)/eval_d.py --model model_d3.pkl --features features_d3.npy \
		--labels $(WORK)/d3/labels --top-features 5

d3: train-d3 fe-d3 eval-d3

partd-all: d1 d2 d3

# ============================================================
# Housekeeping
# ============================================================
clean:
	rm -rf $(WORK)
	rm -f predictions_a.txt weights_a.txt
	rm -f predictions_b.txt weights_b.txt bestlambda_b.txt crossvalidation_errors_b.txt
	rm -f predictions_c.txt
	rm -f model_d1.pkl model_d2.pkl model_d3.pkl
	rm -f features_d1.npy features_d2.npy features_d3.npy