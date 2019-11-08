import json
import javalang
import pandas as pd
import util
from gensim.models.word2vec import Word2Vec
from collections import Counter


def java_method_to_ast(method):
    tokens = javalang.tokenizer.tokenize(method)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


print('Reading data...')

json_files = [open('./data/train.json'), open('./data/valid.json'), open('./data/test.json')]
json_lines = [file.readlines() for file in json_files]
line_count = [len(lines) for lines in json_lines]

print('SIZE of [Training set, Validation set, Test set] =', line_count)

json_lines = json_lines[0] + json_lines[1] + json_lines[2]
json_str = '[' + ','.join(json_lines) + ']'
json_objs = json.loads(json_str)

programs = pd.DataFrame(json_objs)
programs.columns = ['code', 'comment']

print('Data size:', len(programs))

print('Parsing to AST...')

programs['ast'] = programs['code'].apply(java_method_to_ast)
programs['name'] = programs['ast'].apply(lambda root: root.name)
for i in range(len(programs['ast'])):
    programs['ast'][i].name = 'METHOD_NAME'
counter = Counter(programs['name'])
names = sorted(set(programs['name']))
programs['label'] = programs['name'].apply(lambda name: names.index(name) + 1)

print('Name Count:', len(counter))

print('Training word embedding...')

programs['corpus'] = programs['ast'].apply(util.ast2sequence)

w2v = Word2Vec(programs['corpus'], size=128, workers=16, sg=1, min_count=3)  # use w2v[WORD] to get embedding
vocab = w2v.wv.vocab


# transform ASTNode to tree of index in word2vec model
def node_to_index(node):
    result = [vocab[node.token].index if node.token in vocab else len(vocab)]
    for child in node.children:
        result.append(node_to_index(child))
    return result


# transform ast to trees of index in word2vec model
def ast_to_index(ast):
    blocks = []
    util.get_ast_nodes(ast, blocks)
    return [node_to_index(b) for b in blocks]


print('Transforming ast to embedding index tree...')

programs['index_tree'] = programs['ast'].apply(ast_to_index)

programs.pop('name')
programs.pop('code')
programs.pop('ast')
programs.pop('corpus')
programs.pop('comment')

w2v.save('./data/w2v_128')
print("Saved word2vec model at", './data/w2v_128')
programs.to_pickle('./data/programs.pkl')
print("Saved processed data at", './data/programs.pkl')


