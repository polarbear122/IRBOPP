import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

tree = ET.parse("./JAAD-JAAD_2.0/annotations/video_0014.xml")
root = tree.getroot()
for child in root:
    print("tag:", child.tag)
    print("tag:", child.text)
    print("attrib:", child.attrib)
    child.set("set:", "设置属性")
