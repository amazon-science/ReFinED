import os

from pynif import NIFCollection

nif = NIFCollection.load(
    os.path.join("/Users/tayoola/Downloads/wikidata_annotated/opentapioca/data", "istex_test.ttl")
)
# nif = NIFCollection.load(os.path.join('/Users/tayoola/Downloads/wikidata_annotated/opentapioca/data',
#                                       'RSS-500_wd.test.ttl'))

for context in nif.contexts:
    # print(context)
    # print(context.uri)
    # print(dir(context))
    # print(next(context.triples()))
    # print(context.uri)
    # print([dir(x) for x in context.phrases])

    # print([x for x in context.phrases])
    sentence = context.mention
    # print(sentence)
    # print(context.beginIndex)

    for phrase in context.phrases:
        start_index = phrase.beginIndex
        end_index = phrase.endIndex
        length = end_index - start_index
        mention_text = sentence[start_index:end_index]
        wikidata_id = phrase.taIdentRef
        if "http://www.wikidata.org/entity/" in wikidata_id:
            print(mention_text, wikidata_id)
            i += 1

    # print(context.mention)
    # break

print(i)
