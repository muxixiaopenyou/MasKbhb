import os
import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, EsmModel
from utils import load_fasta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def generate_embeddings(seqs, model, tokenizer, device, batch_size):
    model.eval()
    all_emb = []

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                add_special_tokens=True
            ).to(device)

            outputs = model(**inputs)

            last_hidden = outputs.last_hidden_state

            for j, seq in enumerate(batch_seqs):
                emb = last_hidden[j, 1 : len(seq) + 1, :].cpu()

                if 'X' in seq:
                    x_mask = torch.tensor([c == 'X' for c in seq], dtype=torch.bool)
                    emb[x_mask] = 0.0

                all_emb.append(emb)
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {i + len(batch_seqs)}/{len(seqs)} sequences...")

    return all_emb 

def process(fasta_path, out_path, model, tokenizer, device, batch_size):
    logger.info(f"Processing {fasta_path}...")
    ids, seqs, labels = load_fasta(fasta_path)

    if not ids:
        logger.warning(f"No sequences found in {fasta_path}")
        return

    embs_list = generate_embeddings(seqs, model, tokenizer, device, batch_size)

    save_data = {
        'ids': ids,
        'seqs': seqs,
        'labels': torch.tensor(labels, dtype=torch.long),
        'embeddings': embs_list 
    }
    
    torch.save(save_data, out_path)
    logger.info(f"Saved: {out_path} (Total: {len(embs_list)} embeddings)")

def main():
    data_dir = 'data'
    # here, change the path to your local model if you have downloaded it
    # otherwise it will download from Hugging Face
    local_model_path = './esm2_model'
    online_model_name = 'facebook/esm2_t33_650M_UR50D'
    
    if os.path.exists(local_model_path):
        model_name_or_path = local_model_path
        logger.info(f"Loading local model from {local_model_path}")
    else:
        model_name_or_path = online_model_name
        logger.info(f"Local model not found. Downloading from Hugging Face: {model_name_or_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = EsmModel.from_pretrained(model_name_or_path).to(device)

    model.eval()

    for split in ['train', 'test']:
        fasta_file = os.path.join(data_dir, f'{split}.fasta')
        output_file = os.path.join(data_dir, f'{split}_embedding.pt')
    
        process(
            fasta_file,
            output_file,
            model, tokenizer, device, batch_size=4
        )

if __name__ == '__main__':
    main()