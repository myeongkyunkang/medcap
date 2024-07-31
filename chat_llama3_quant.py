import os

import fire
import numpy as np
import pandas as pd


def main(
        ckpt_dir='',
        temperature=0.0001,
        top_p=0.9,
        max_new_tokens=200,
        bit=8,
        gpu=(0, 1),
        exp='chat',
        csv_path=None,
):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpu]) if isinstance(gpu, tuple) else str(gpu)

    # import after CUDA_VISIBLE_DEVICES declared
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

    # load model
    if bit == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
    elif bit == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    else:
        raise ValueError('Invalid bit:', bit)

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        quantization_config=bnb_config,
        device_map='auto',
        local_files_only=True,
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)

    # create generation_pipe
    generation_pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
    )

    # set terminators
    terminators = [
        generation_pipe.tokenizer.eos_token_id,
        generation_pipe.tokenizer.convert_tokens_to_ids('<|eot_id|>')
    ]

    if exp == 'chat':
        while True:
            instruction = input('Inst: ').strip()
            if instruction == 'q':
                break
            elif len(instruction) == 0:
                continue

            dialogs = [{'role': 'user', 'content': instruction}, ]
            outputs = generation_pipe(
                dialogs,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
            )
            generated_text = outputs[0]['generated_text'][-1]['content']
            print(generated_text)
            print('=' * 20)
    elif exp == 'eval_vqa':
        # read csv
        answer_dict = pd.read_csv(csv_path)

        dialogs_list = []
        for inst, out, gen in zip(answer_dict['instruction'], answer_dict['output'], answer_dict['generated']):
            vqa_question = inst.split('<|eot_id|><|start_header_id|>user<|end_header_id|>')[-1].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip()
            question = 'Respond with only RIGHT or WRONG to indicate whether we predicted it correctly.\n\n'
            question += f'Question: {vqa_question}\nOur answer: {gen}\nCorrect answer: {out}'
            dialogs_list.append([{'role': 'user', 'content': question}])

        correct_list = []
        out_dict = {'image': [], 'instruction': [], 'generated': []}
        for dialogs, image in zip(dialogs_list, answer_dict['image']):
            outputs = generation_pipe(
                dialogs,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
            )
            generated_text = outputs[0]['generated_text'][-1]['content']
            print(dialogs[0]['content'])
            print(generated_text)
            print('=' * 20)

            generated_text_lower = generated_text.lower()

            if 'right' in generated_text_lower:
                correct_list.append(1)
            elif 'wrong' in generated_text_lower:
                correct_list.append(0)
            else:
                correct_list.append(0)  # quite rare

            out_dict['image'].append(image)
            out_dict['instruction'].append(dialogs[0]['content'])
            out_dict['generated'].append(generated_text)

        pd.DataFrame(out_dict).to_csv(csv_path.replace('.csv', f'_70B_{bit}bit_out.csv'), index=False, encoding='utf-8-sig')
        with open(csv_path.replace('.csv', f'_70B_{bit}bit_acc.csv'), 'wt') as f:
            f.write(f'{np.mean(correct_list)}\n')


if __name__ == '__main__':
    fire.Fire(main)
