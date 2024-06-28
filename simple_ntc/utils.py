def read_text(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines() # 문서 전체 라인 읽기
        # print(lines)
        snos, labels, texts = [], [], [] # 라벨, 텍스트 리스트 선언

        for line in lines:
            if line.strip() != '':
                # print(line)
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                # sno, label, text = line.strip().split('\t')
                label, text = line.strip().split('\t')
                # snos += [sno]

                labels += [label]
                texts += [text]

    return labels, texts # 일련번호, 라벨, 텍스트 리스트 return

def read_test_data(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines() # 문서 전체 라인 읽기
        # 라벨, 텍스트 리스트 선언
        labels, texts =[],[]

        for line in lines:
            if line.strip() != '':
                # print(line)
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                # sno, label, text, gram, meta_0_v, meta_1_v, meta_0_r, meta_1_r, item = line.strip().split('\t')

                label, text = line.strip().split('\t')
                # snos += [sno]
                labels += [label]
                texts += [text]
                # grams += [gram]
                # meta_0_values += [meta_0_v]
                # meta_1_values += [meta_1_v]
                # meta_0_results += [meta_0_r]
                # meta_1_results += [meta_1_r]
                # items += [item]

    # return snos, labels, texts, grams, meta_0_values, meta_1_values, meta_0_results, meta_1_results, items # 일련번호, 라벨, 텍스트 리스트 return
    return labels, texts



def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
