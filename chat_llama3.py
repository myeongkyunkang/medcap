from typing import List, Optional

import fire
import numpy as np
import pandas as pd
from llama import Dialog, Llama


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_seq_len: int = 200,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        exp='chat',
        csv_path=None,
):
    # load llama
    generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)

    if exp == 'chat':
        while True:
            instruction = input('Inst: ').strip()
            if instruction == 'q':
                break
            elif len(instruction) == 0:
                continue
            dialogs: List[Dialog] = [[{"role": "user", "content": instruction}], ]
            results = generator.chat_completion(dialogs, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            for result in results:
                print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
                print('=' * 20)
    elif exp == 'eval_vqa':
        # read csv
        answer_dict = pd.read_csv(csv_path)

        dialogs: List[Dialog] = []
        for inst, out, gen in zip(answer_dict['instruction'], answer_dict['output'], answer_dict['generated']):
            vqa_question = inst.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[-1].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip()
            question = "Respond with only RIGHT or WRONG to indicate whether we predicted it correctly.\n\n"
            question += f"Question: {vqa_question}\nOur answer: {gen}\nCorrect answer: {out}"
            dialogs.append([{"role": "user", "content": question}])

        correct_list = []
        out_dict = {'image': [], 'instruction': [], 'generated': []}
        for d_i in range(0, len(dialogs), max_batch_size):
            _dialogs = dialogs[d_i:d_i + max_batch_size]
            _images = answer_dict['image'][d_i:d_i + max_batch_size]

            results = generator.chat_completion(_dialogs, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

            print(_dialogs[0][0]['content'])
            print(results[0]['generation']['content'])
            print('=' * 20)

            for img, diag, res in zip(_images, _dialogs, results):
                generated_text = res['generation']['content']
                generated_text_lower = generated_text.lower()

                if 'right' in generated_text_lower:
                    correct_list.append(1)
                elif 'wrong' in generated_text_lower:
                    correct_list.append(0)
                else:
                    correct_list.append(0)  # quite rare

                out_dict['image'].append(img)
                out_dict['instruction'].append(diag[0]['content'])
                out_dict['generated'].append(generated_text)

        pd.DataFrame(out_dict).to_csv(csv_path.replace('.csv', '_70B_out.csv'), index=False, encoding='utf-8-sig')
        with open(csv_path.replace('.csv', '_70B_acc.csv'), 'wt') as f:
            f.write(f"{np.mean(correct_list)}\n")


if __name__ == "__main__":
    fire.Fire(main)
