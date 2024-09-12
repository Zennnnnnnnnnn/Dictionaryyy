import streamlit as st
import re
import xml.etree.ElementTree as et

# Hàm tìm vị trí của từ trong từ điển
def findword(nhap_1_tu, root):
    kq = root.findall(".//runhd")
    for tu in kq:
        if tu.text == nhap_1_tu:
            print("word = ", tu.text)
            return tu
    return None

# Hàm tìm cha của 1 node
def findfather(ptu, root):
    parent = None
    for i in root.iter():  # Duyệt qua tất cả các node từ root xuống
        for child in i:
            if child == ptu:
                parent = i
                break
        if parent is not None:
            break
    return parent

# Functions for EEV Dictionary Processing
def textprocess(chuoi):
    newchuoi = chuoi.strip()
    kytudb = r'[=~!@#$%^&*<>?:{}|\\\[\]/]'
    newchuoi = re.sub(r'/[^/]+/', "", newchuoi)
    newchuoi = re.sub(kytudb, "", newchuoi)
    newchuoi = re.sub(r'\s{2,}', ' ', newchuoi)
    newchuoi = newchuoi.replace(" ,", ",")
    return newchuoi

def text_outside_children(nodecha):
  full_text = ''.join(nodecha.itertext())   # Get all text within the parent node
  #print("** full_text = ", full_text) ; 
  txt_v_s_srf=nodecha.find(".//txt_v_s_srf")
  txt_v_s_srf_texts=''.join(txt_v_s_srf.itertext()) ;  print ("txt_v_s_srf = ", txt_v_s_srf_texts) # chuỗi các text có dưới tag <txt_v_s_srf>
  full_text=full_text.replace(txt_v_s_srf_texts,"") ;# loại các text có trong <txt_v_s_srf>: text, tail, con-cháu
  
  symbol=nodecha.find(".//txt_v_s_srf")
  symbol_texts=''.join(symbol.itertext())  # chuỗi các text có dưới tag <symbol>
  full_text=full_text.replace(symbol_texts,"").replace("","")  # loại các text có trong <symbol>: text, tail, con-cháu
  #print("** full_text Sau khi loại nghĩa việt = ",full_text)  # loại tag nghĩa tiếng việt ra khỏi chuỗi
  # Remove text inside any child nodes
  for child in nodecha:
    if child.tag == nodecha.tag: continue  # bỏ qua nodecha không xét
    if child.tag == "dh": continue      # Thông in trong node <dh> là lấy, trong z là lấy, không có bỏ ra
    if child.tag == "z": continue
    #print("node con là = ", child.tag)
    child_text = ''.join(child.itertext()) ; print("** child_text = ", child_text.strip())
    if child_text ==" " : continue
    full_text = full_text.replace(child_text, '').replace("‘","").replace("’","")
    
  # Clean up and return the result
  return ' '.join(full_text.split()).strip()
    
def extract_example_text(node):
    """
    Extracts and processes example text from a given XML node.
    Handles nested elements like <gl>, <xr>, and <xh> correctly.
    """
    example_text = ''
    
    for elem in node.iter():
        if elem.tag == 'x':
            # Process <x> and its content
            x_text = ''.join(textprocess(e.text.strip()) if e.text else '' for e in elem.iter())
            example_text += x_text
        
        elif elem.tag == 'gl':
            # Process <gl> and its content
            gl_text = ''.join(textprocess(e.text.strip()) if e.text else '' for e in elem.iter())
            example_text += f" ({gl_text})"
        
        elif elem.tag == 'xr':
            # Process <xr> and its content
            xr_text = ''.join(textprocess(e.text.strip()) if e.text else '' for e in elem.iter())
            example_text += f" ({xr_text})"
        
        elif elem.tag == 'xh':
            # Process <xh> and its content
            xh_text = textprocess(elem.text.strip()) if elem.text else ''
            example_text += f"{xh_text}"
    
    return example_text.strip()

    
def meaningex(d_ud, root):
    word_type = d_ud.find(".//p-g")
    if word_type is not None:
        type_text = textprocess(word_type.text.strip()) if word_type.text else ""
    else:
        type_text = ""

    em = ""
    if d_ud.tag in ["d", "ud"]:
        outside_text = text_outside_children(d_ud).strip()
        if outside_text:
            em = textprocess(outside_text)
        else:
            for dud in d_ud.iter():
                if dud.tag not in ["d", "ud", 'txt_v_s_srf', 'symbol', "space", 'meaning', "z_xr", "i", "z", "xr", "xh", "cap_in_xh"]:
                    text = textprocess(dud.text.strip()) if dud.text else ""
                    tail = textprocess(dud.tail.strip()) if dud.tail else ""
                    em += text + tail
    elif d_ud.tag in ["xr"]:
        for dud in d_ud.iter():
            text = textprocess(dud.text.strip()) if dud.text else ""
            tail = textprocess(dud.tail.strip()) if dud.tail else ""
            em += text + tail

    vm = ""
    txt_v_s_srf = d_ud.find(".//txt_v_s_srf")
    if txt_v_s_srf is not None:
        for txt in txt_v_s_srf.iter():
            text = textprocess(txt.text.strip()) if txt.text else ""
            tail = textprocess(txt.tail.strip()) if txt.tail else ""
            vm = text + tail

    ex = []
    # Extract examples
    vidu = findfather(d_ud, root).findall(".//x")
    for vd in vidu:
        example_text = extract_example_text(vd)
        if example_text:
            ex.append(example_text)

    return em, vm, ex

def thongtin1tu(word, root):
    thongtin = {
        'pronunciation': [],
        'word_type': [],
        'meanings': []
    }

    runhd = findword(word, root)
    if runhd is None:
        return thongtin

    father = findfather(runhd, root)
    grand = findfather(father, root)
    greatgrand = findfather(grand, root)

    phienam = father.findall(".//i")
    for j in phienam:
        thongtin['pronunciation'].append(j.text)

    p_g = greatgrand.findall(".//p-g")
    if not p_g:
        word_type = greatgrand.find(".//z_p")
        thongtin['word_type'].append(word_type.text.strip() if word_type is not None else "Không xác định")

        n_g = greatgrand.findall(".//n-g")
        if not n_g:
            d_ud = greatgrand.find(".//d") or greatgrand.find(".//ud") or greatgrand.find(".//xr")
            if d_ud:
                em, vm, ex = meaningex(d_ud, root)
                meaning_info = {
                    'meanings_english': em,
                    'meanings_vietnamese': vm,
                    'examples': ex,
                    'word_type': thongtin['word_type'][0]
                }
                thongtin['meanings'].append(meaning_info)
        else:
            for ng in n_g:
                if findfather(ng, root).tag not in ['pv-g', 'id-g']:
                    zn = ng.find(".//zn")
                    meaning_info = {
                        'meanings_english': zn.text.strip() if zn is not None else "Không xác định",
                        'meanings_vietnamese': None,
                        'examples': [],
                        'word_type': thongtin['word_type'][0]
                    }
                    d_ud = ng.find(".//d") or ng.find(".//ud") or ng.find(".//xr")
                    if d_ud:
                        em, vm, ex = meaningex(d_ud, root)
                        meaning_info['meanings_english'] = em
                        meaning_info['meanings_vietnamese'] = vm
                        meaning_info['examples'].extend(ex)
                    thongtin['meanings'].append(meaning_info)
    else:
        for pg in p_g:
            word_type = pg.find(".//z_p_in_p-g")
            current_word_type = word_type.text.strip() if word_type is not None else "Không xác định"
            thongtin['word_type'].append(current_word_type)

            n_g = pg.findall(".//n-g")
            if not n_g:
                d_ud = pg.find(".//d") or pg.find(".//ud") or pg.find(".//xr")
                if d_ud:
                    em, vm, ex = meaningex(d_ud, root)
                    meaning_info = {
                        'meanings_english': em,
                        'meanings_vietnamese': vm,
                        'examples': ex,
                        'word_type': current_word_type
                    }
                    thongtin['meanings'].append(meaning_info)
            else:
                for ng in n_g:
                    if findfather(ng, root).tag not in ['pv-g', 'id-g']:
                        zn = ng.find(".//zn")
                        meaning_info = {
                            'meanings_english': zn.text.strip() if zn is not None else "Không xác định",
                            'meanings_vietnamese': None,
                            'examples': [],
                            'word_type': current_word_type
                        }
                        d_ud = ng.find(".//d") or ng.find(".//ud") or ng.find(".//xr")
                        if d_ud:
                            em, vm, ex = meaningex(d_ud, root)
                            meaning_info['meanings_english'] = em
                            meaning_info['meanings_vietnamese'] = vm
                            meaning_info['examples'].extend(ex)
                        thongtin['meanings'].append(meaning_info)

    back_meanings = []
    for index, meaning in enumerate(thongtin['meanings'], start=1):
        meaning_info = {
            "Keyword": f"{index}",
            "Description": meaning['meanings_english'],
            "Meaning_Vietnamese": meaning['meanings_vietnamese'],
            "Examples": meaning['examples'],
            "Word_Type": meaning['word_type'],
            "Pronunciation": ", ".join(thongtin['pronunciation'])
        }
        back_meanings.append(meaning_info)

    return back_meanings
