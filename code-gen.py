
import subprocess

with open('dgemm-blocked.c', 'r') as f:
    source_lines = f.readlines()

best_ratio = 0
config = [0, 0]

# for BLOCK_SIZE1 in range(20, 41):
#     for BLOCK_SIZE2 in range(90, 201):
for BLOCK_SIZE2 in range(90, 201):
    for BLOCK_SIZE1 in range(20, 70):
        tmp = source_lines[26].strip().split()
        tmp[2] = str(BLOCK_SIZE1)
        source_lines[26] = ' '.join(tmp) + '\n'

        tmp = source_lines[27].strip().split()
        tmp[2] = str(BLOCK_SIZE2)
        source_lines[27] = ' '.join(tmp) + '\n'

        # print(''.join(lines))
        with open('dgemm-blocked.c', 'w') as f:
            f.write(''.join(source_lines))

        if subprocess.call('make', stdout=subprocess.DEVNULL) != 0:
            exit()

        if subprocess.call('./benchmark-blocked', stdout=subprocess.DEVNULL) != 0:
            exit()

        with open('result.txt', 'r') as f:
            result_lines = f.readlines()

        ratio = float(result_lines[5].strip().split()[-1])

        if ratio > best_ratio:
            best_ratio = ratio
            config = [BLOCK_SIZE1, BLOCK_SIZE2]
        print('SIZE1 = %d, SIZE2 = %d, Ratio = %f' % (BLOCK_SIZE1, BLOCK_SIZE2, ratio))


print ('Best Ratio = %f, Config = [%d, %d]' % (best_ratio, config[0], config[1]))