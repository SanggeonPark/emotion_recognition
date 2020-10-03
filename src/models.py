from switcher import l2i_switcher
import declxml as xml

# Dictionary Structure
xml_preprocessor = xml.dictionary('annotation', [
    xml.string('filename'),
    xml.dictionary('size', [
        xml.integer('width'),
        xml.integer('height'),
        xml.integer('depth')
    ]),
    xml.array(xml.dictionary('object', [
        xml.string('name'),
        xml.dictionary('bndbox', [
            xml.integer('xmin'),
            xml.integer('ymin'),
            xml.integer('xmax'),
            xml.integer('ymax')
        ])
    ]), alias='objects')
])

def dictionary_from_xml_file_url(xml_file_url):
    return xml.parse_from_file(xml_preprocessor, xml_file_url)

class TrainData:
    def __init__(self, data, label):
        self.data = data
        switcher = l2i_switcher.get(label, lambda: "Invalid label")
        self.label_number = switcher()
