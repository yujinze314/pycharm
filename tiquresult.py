with open('all_output.txt', 'r', encoding='utf-8') as fin, open('results_only.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        if line.startswith('Result:'):
            fout.write(line)