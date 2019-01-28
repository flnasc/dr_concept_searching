"""Utilities file. Takes a given xml document and adds an attribute ID 
to each of its paragraph tags"""
import xml.etree.ElementTree

def load(filename):
	et = xml.etree.ElementTree.parse(filename)
	p_tags = et.iter(tag="{http://www.tei-c.org/ns/1.0}p")
	id_num = 0
	for tag in p_tags:
		tag.set("ID",str(id_num))
		id_num += 1
	et.write(filename)


if __name__ == "__main__":
	load("../data/oneself_as_another.xml")