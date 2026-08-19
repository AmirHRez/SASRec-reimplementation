import os
import time
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm

from model import SASRec
from utils import *


@dataclass
class Config:
    dataset: str
    train_dir: str
    batch_size: int = 128
    lr: float = 0.001
    maxlen: int = 200
    hidden_units: int = 50
    num_blocks: int = 2
    num_epochs: int = 1000
    num_heads: int = 1
    dropout_rate: float = 0.2
    l2_emb: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    inference_only: bool = False
    state_dict_path: str | None = None
    norm_first: bool = False


def save_checkpoint(path, model, optimizer, epoch, args):
    tmp_path = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": asdict(args),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


args = Config(dataset="ml-1m", train_dir="default")
# remove `state_dict_path` when starting new
args.state_dict_path = "ml-1m_default/latest.pth"
if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)

with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f:
    f.write("\n".join(f"{k},{v}" for k, v in sorted(asdict(args).items())))
f.close()

if __name__ == "__main__":
    u2i_index, i2u_index = build_index(args.dataset)

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )
    model = SASRec(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # ignore

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    model.train()

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            ckpt = torch.load(
                args.state_dict_path, map_location=torch.device(args.device)
            )
            model.load_state_dict(ckpt["model_state_dict"])
            adam_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            epoch_start_idx = ckpt["epoch"] + 1
        except:
            print(f"No checkpoint file at {args.state_dict_path}")

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)

        print(
            f"test (NDCG@10: {t_test['ndcg']:.4f}, HR@10: {t_test['hr']:.4f}, "
            f"Precision@10: {t_test['precision']:.4f}, Recall@10: {t_test['recall']:.4f}, "
            f"MRR: {t_test['mrr']:.4f})"
        )

    bce_criterion = torch.nn.BCEWithLogitsLoss()

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    folder = args.dataset + "_" + args.train_dir

    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1), desc="epochs"):
        if args.inference_only:
            break  # decrease identition

        epoch_loss = 0.0
        for step in tqdm(range(num_batch), desc=f"epoch {epoch}", leave=False):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = (
                torch.ones(pos_logits.shape, device=args.device),
                torch.zeros(neg_logits.shape, device=args.device),
            )
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param**2)

            loss.backward()
            adam_optimizer.step()
            epoch_loss += loss.item()

        tqdm.write(f"epoch {epoch}: avg loss {epoch_loss / num_batch:.4f}")

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            tqdm.write("Evaluating", end="")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            tqdm.write(
                f"epoch:{epoch}, time: {T}(s), valid (NDCG@10: {t_valid['ndcg']:.4f}, HR@10: {t_valid['hr']:.4f}), "
                f"test (NDCG@10:{t_test['ndcg']:.4f}, HR@10: {t_test['hr']:.4f})"
            )

            if (
                t_valid["ndcg"] > best_val_ndcg
                or t_valid["hr"] > best_val_hr
                or t_test["ndcg"] > best_test_ndcg
                or t_test["hr"] > best_test_hr
            ):
                best_val_ndcg = max(t_valid["ndcg"], best_val_ndcg)
                best_val_hr = max(t_valid["hr"], best_val_hr)
                best_test_ndcg = max(t_test["ndcg"], best_test_ndcg)
                best_test_hr = max(t_test["hr"], best_test_hr)

                fname = f"SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth"
                torch.save(model.state_dict(), os.path.join(folder, fname))

            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            fname = f"SASRec.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth"
            torch.save(model.state_dict(), os.path.join(folder, fname))

        # save checkpoint every 5 epoch
        if epoch % 5 == 0:
            save_checkpoint(
                os.path.join(folder, "latest.pth"), model, adam_optimizer, epoch, args
            )

    f.close()
    sampler.close()
    print("Done")
