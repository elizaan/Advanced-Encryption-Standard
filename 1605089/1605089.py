#!/usr/bin/env python
# coding: utf-8

# # Importing necessary modules

# In[20]:


from __future__ import print_function
import cloudmersive_convert_api_client
from cloudmersive_convert_api_client.rest import ApiException
from pprint import pprint
from PIL import Image
import io
import time
from time import time
import numpy as np
import PyPDF2
from fpdf import FPDF 
from BitVector import *


Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

Mixer = [
    [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01")],
    [BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02"), BitVector(hexstring="03")],
    [BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01"), BitVector(hexstring="02")]
]

InvMixer = [
    [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")],
    [BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D")],
    [BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E"), BitVector(hexstring="0B")],
    [BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09"), BitVector(hexstring="0E")]
]


# # Converting to HexaDecimal

# In[14]:



def convertingToHex(k, frst, c):
 
    # Initialize final String
    hexa = ""
    if(c == 2):
        
        for i in range(len(k)):
            ch = k[i]
            #in1 = ord(ch)
            y = int(ch)
            part = hex(y)
            frst[i] = part
            hexa = hexa + part
    else:
        #hexa = ""
    
        for i in range(len(k)):

            ch = k[i]

            in1 = ord(ch)

            part = hex(in1).lstrip("0x").rstrip("L")
           # print(part)
            frst[i] = part

    #         for i in range(0,4):
    #             for j in range(0,4):
    #                 first[j][i] = part

    #         np.append(first,(part))



            hexa += part

    #print(type(hexa))

    return hexa,frst


# # Converting from HextoASCIItoText 

# In[15]:



def convertingTotext(stat, c):
    if c == 2:
        image_data = bytes.fromhex(stat)
        image = Image.open(io.BytesIO(image_data))
        image.show()
        
        
    else:
        
        bytes_object = bytes.fromhex(stat)
        ascii_string = bytes_object.decode("ASCII")
 
        return ascii_string



# # Calculating g(w)

# In[16]:


def calculateG(lst,roundConstant):
    lst2=[0,0,0,0]
    #step1: circular byte shift
    lst = np.asarray(lst)
    lst = np.roll(lst, -1)
    #step 2: Byte substitution
    for i in range(0,len(lst)):
        
        b = BitVector(hexstring=lst[i])
        int_val = b.intValue()
        s = Sbox[int_val]
        s = BitVector(intVal=s, size=8)
        lst[i] = s.get_bitvector_in_hex()
        #step 3: Adding round constant
    array1 =[int(value, 16) for value in lst]
    array2 =[int(value, 16) for value in roundConstant]
#     print(array1)
#     print(array2)
    for i in range(0,len(array1)):
        lst2[i] = array1[i] ^ array2[i]
        lst2[i] = BitVector(intVal= lst2[i], size = 8).get_bitvector_in_hex()
#     print(lst)
    return lst2


# # Taking input and converting the key and text into ASCII value 

# In[17]:




first = [0 for i in range(0,16)]
# second = [0 for i in range(16)]
key = input("Enter the key: " )

        
if(len(key) > 16):
    for i in range(0,16):
        key[i] = key[i]


hex_key,first= convertingToHex(key, first,0)
#print(len(hex_key))
if(len(hex_key) < 32):
    for d in range(len(hex_key),32,2):
        hex_key = hex_key+ BitVector(intVal = 0, size = 8).get_bitvector_in_hex()
    for d in range(0,len(first)):
        if(first[d] == ''):
            first[d] = "00"

#print(len(hex_key))
print(hex_key)

# np.transpose(first)
first= np.reshape(first, (4,4))
first = np.transpose(first)

print(first)


# # Saving w0,w1,w2,w3

# In[18]:


w=[[0 for i in range(0,4)] for j in range(0,4)]
w=[[first[0][0],first[1][0],first[2][0],first[3][0]],[first[0][1],first[1][1],first[2][1],first[3][1]],[first[0][2],first[1][2],first[2][2],first[3][2]],[first[0][3],first[1][3],first[2][3],first[3][3]]]
# w0 = np.asarray(w.copy())
# w0 = np.transpose(w0)
# print(w0)


# # Key Scheduling

# In[21]:


r=[]
g = [0,0,0,0]
lst3 = [0,0,0,0]
ktime = 0
start = time()
for i in range(0,40,4):
        #print(i)
        if(i==0):
            r.append([BitVector(intVal=0x01, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex()])
            r[0] = np.asarray(r[0])
        #elif (i>0 and int(r[len(r)-1][0],16)< int('0x80',16)):
        elif (i>0):
            AES_modulus = BitVector(bitstring='100011011')

            bv1 = BitVector(hexstring="02")
            bv2 = BitVector(hexstring=r[len(r)-1][0])
            bv3 = bv1.gf_multiply_modular(bv2, AES_modulus, 8)
            r.append([bv3.get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex()])
            r[len(r)-1] = np.asarray(r[len(r)-1])
            # This check should be used but idk why it gives error,if I use it,
            #nothing matches with sir from round 9, then it gives index error in 
            #generating g, then round 10 doesn't occur
    #     elif (i>0 and int(r[len(r)-1][0],16)>= int('0x80',16)): 

    #         AES_modulus = BitVector(bitstring='100011011')
    #         print("stupid :)")
    #         bv1 = BitVector(hexstring="02")
    #         bv2 = BitVector(hexstring=r[len(r)-1][0])
    #         bv3 = bv1.gf_multiply_modular(bv2, AES_modulus, 8)
    #         x = int(bv3.get_bitvector_in_hex(),16) ^ int('0x11B',16)
    #         x = hex(x).lstrip("0x").rstrip("L")
    #         r.append([x,BitVector(intVal=0x00, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex(),BitVector(intVal=0x00, size=8).get_bitvector_in_hex()])
    #         r[len(r)-1] = np.asarray(r[len(r)-1])
        g =  calculateG(w[i+3].copy(),r[len(r)-1]) 

        array1 =[int(value, 16) for value in g]
        array2 =[int(value, 16) for value in w[i].copy()]

        for k in range(0,len(array1)):
            lst3[k] = array1[k] ^ array2[k]
            lst3[k] = BitVector(intVal= lst3[k], size = 8).get_bitvector_in_hex()
        w.append([lst3[0],lst3[1],lst3[2],lst3[3]])  


        array1 =[int(value, 16) for value in w[len(w)-1].copy()]
        array2 =[int(value, 16) for value in w[i+1].copy()]

        for j in range(0,len(array1)):
            lst3[j] = array1[j] ^ array2[j]
            lst3[j] = BitVector(intVal= lst3[j], size = 8).get_bitvector_in_hex()
            #print(lst3[k])
        w.append([lst3[0],lst3[1],lst3[2],lst3[3]]) 

        array1 =[int(value, 16) for value in w[len(w)-1].copy()]
        array2 =[int(value, 16) for value in w[i+2].copy()]

        for h in range(0,len(array1)):
            lst3[h] = array1[h] ^ array2[h]
            lst3[h] = BitVector(intVal= lst3[h], size = 8).get_bitvector_in_hex()
        w.append([lst3[0],lst3[1],lst3[2],lst3[3]]) 

        array1 =[int(value, 16) for value in w[len(w)-1].copy()]
        array2 =[int(value, 16) for value in w[i+3].copy()]

        for d in range(0,len(array1)):
            lst3[d] = array1[d] ^ array2[d]
            lst3[d] = BitVector(intVal= lst3[d], size = 8).get_bitvector_in_hex()
        w.append([lst3[0],lst3[1],lst3[2],lst3[3]])

end = time()
ktime = end - start


# # Add roundkey: round 0, ciphering and deciphering

# In[23]:


cipher = ""
decipher = ""
text = ""
choice = 0
htext=""
etime = 0
dtime = 0

choice = int(input("Enter what you want to read? 1. pdf, 2.image 3.input text 4. text file/python file: "))
print(choice)

if (choice == 1):
    filename = input("Enter filename: ")
    pdFileobj = open(filename, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdFileobj)
    pageObj = pdfReader.getPage(0)
    text = pageObj.extractText()

if (choice==2):
    image = input("Enter the image file: ")
    with open (image, 'rb') as img:
        f = img.read()
        text = bytearray(f)
    #print(type(text))

    
elif (choice == 3):
    
    text = input("Enter the text: ")

elif (choice==4):
    filename = input("Enter filename: ")
    pdf = FPDF()    
   
    # Add a page 
    pdf.add_page() 

    # set style and size of font  
    # that you want in the pdf 
    pdf.set_font("Arial", size = 15) 

    # open the text file in read mode 
    fil = open(filename, "r") 

    # insert the texts in pdf 
    for x in fil: 
        pdf.cell(200, 10, txt = x, ln = 1, align = 'C')

    pdf.output("demo.pdf") 
    pdFileobj = open("demo.pdf", 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdFileobj)
    pageObj = pdfReader.getPage(0)
    text = pageObj.extractText()

if(len(text)<16):
    for f in range(len(text),16):
        text = text+" "
if(len(text)>16 and len(text)%16 !=0):
    for f in range(len(text)%16, 16):
        if(choice == 2):
            text.append(20)
        else:
            text = text+" "


for y in range(0,len(text),16):
    second = [0 for i in range(16)]
    hex_text,second= convertingToHex(text[y:y+16], second, choice)
    htext = htext + hex_text
    #print(hex_text)
    second = np.reshape(second, (4,4))
    second = np.transpose(second)
    
    start = time()
    #step0: Adding roundkey 0 
    w0 = np.array(w.copy()[0:4]).T.tolist()  
    state0 = [[0 for i in range(0,4)] for j in range(0,4)]
    arr1 =[[0 for t in range(len(state0))] for u in range(len(state0))]
    arr2 =[[0 for t in range(len(w0))] for u in range(len(w0))]


    for m in range(0,4):
        for n in range(0,4):
            #print(int(second[i][j], 16))
            arr1[m][n] = int(second[m][n].copy(),16)
            arr2[m][n] = int(w0.copy()[m][n], 16)
    



    for m in range(0,len(arr1)):
        for n in range(0,len(arr1)):
            state0[m][n] = arr1[m][n] ^ arr2[m][n]
            state0[m][n] = BitVector(intVal= state0[m][n], size = 8).get_bitvector_in_hex()
    end = time()
    etime = etime + (end-start)
    
    #####hereeee##
    

    
    for i in range(0,40,4):
        #print(i)

        #print(state0)
        start1 = time()
        #step1: substituting Sbox Entry

        for c in range(0,4):
            for a  in range(0,4):
                b = BitVector(hexstring=state0[c][a])
                int_val = b.intValue()
                s = Sbox[int_val]
                s = BitVector(intVal=s, size=8)
                state0[c][a] = s.get_bitvector_in_hex()
        #print(state0)

        #step2: shifting rows as their row number

        state0[1:2] = np.roll(state0[1:2], -1).tolist()
        state0[2:3] = np.roll(state0[2:3], -2).tolist()
        state0[3:4] = np.roll(state0[3:4], -3).tolist()
        #print(state0)

        #step3: Mix columns
        if(i!=36):
            res =[[0 for t in range(len(Mixer))] for u in range(len(state0))]
            AES_modulus = BitVector(bitstring='100011011')
            for p in range (len(Mixer)):
                for e in range(len(state0[0])):
                    for l in range(len(state0)):
                        v1 = Mixer[p][l]
                        v2 = BitVector(hexstring = state0[l][e])
                        m = v1.gf_multiply_modular(v2,AES_modulus,8)
                        res[p][e] = res[p][e] ^ int(m.get_bitvector_in_hex(),16)


            for p in range(len(res)):
                for e in range(len(res[p])):
                     state0[p][e] = BitVector(intVal= res[p][e], size = 8).get_bitvector_in_hex()

        #print(state0)



        #step4: adding next round key

        #w0 = np.asarray(w.copy())
        w0 = np.array(w.copy()[i+4:i+8]).T.tolist()
        #print(w0)
        arr1 =[[0 for t in range(len(state0))] for u in range(len(state0))]
        arr2 =[[0 for t in range(len(w0))] for u in range(len(w0))]



        for xxx in range(0,4):
            for yyy in range(0,4):
                #print(int(second[i][j], 16))
                arr1[xxx][yyy] = int(state0.copy()[xxx][yyy], 16)
                arr2[xxx][yyy] = int(w0.copy()[xxx][yyy], 16)
    


    

        for mm in range(0,len(arr1)):
            for nn in range(0,len(arr2)):
                state0[mm][nn] = arr1[mm][nn] ^ arr2[mm][nn] 
                #print(state0[mm][nn])
                state0[mm][nn] = BitVector(intVal= int(state0[mm][nn]), size = 8).get_bitvector_in_hex()

        #print(state0)
        end1 = time()
        etime = etime + (end1-start1)
    # Ciphertexting is done!
    state0 = np.array(state0.copy()).T.tolist()
    for i in range(len(state0)):
        for j in range(len(state0[0])):
            cipher = cipher+state0[i][j]

  
    #Decipher starts here
    start3 = time()
    state1 = np.array(state0.copy()).T.tolist()
    
    # Adding roundkey w(40,43)
    
    w1 = np.array(w.copy()[40:44]).T.tolist()  
    #state0 = [[0 for i in range(0,4)] for j in range(0,4)]
    arr1 =[[0 for t in range(len(state1))] for u in range(len(state1))]
    arr2 =[[0 for t in range(len(w1))] for u in range(len(w1))]


    for m in range(0,4):
        for n in range(0,4):
            #print(int(second[i][j], 16))
            arr1[m][n] = int(state1.copy()[m][n],16)
            arr2[m][n] = int(w1.copy()[m][n], 16)
    



    for m in range(0,len(arr1)):
        for n in range(0,len(arr1)):
            state1[m][n] = arr1[m][n] ^ arr2[m][n]
            state1[m][n] = BitVector(intVal= state1[m][n], size = 8).get_bitvector_in_hex()
    
    #Going through other steps
    end3 = time()
    dtime = dtime + (end3-start3)
    
    lst3 = [0,0,0,0]

    for i in range(0,40,4):
#         print(i)
#         print(state1)
        start4 = time()

        #step1: shifting rows inversly as their row number

        state1[1:2] = np.roll(state1[1:2], 1).tolist()
        state1[2:3] = np.roll(state1[2:3], 2).tolist()
        state1[3:4] = np.roll(state1[3:4], 3).tolist()
        #print(state1)

        #step2: substituting inverse Sbox Entry

        for c in range(0,4):
            for a  in range(0,4):
                b = BitVector(hexstring=state1[c][a])
                int_val = b.intValue()
                s = InvSbox[int_val]
                s = BitVector(intVal=s, size=8)
                state1[c][a] = s.get_bitvector_in_hex()
        #print(state1)

         #step3: adding previous round key

        #w0 = np.asarray(w.copy())
        w1 = np.array(w.copy()[36-i:40-i]).T.tolist()
        #print(w1)
        arr1 =[[0 for t in range(len(state1))] for u in range(len(state1))]
        arr2 =[[0 for t in range(len(w1))] for u in range(len(w1))]



        for xxx in range(0,4):
            for yyy in range(0,4):
                #print(int(second[i][j], 16))
                arr1[xxx][yyy] = int(state1.copy()[xxx][yyy], 16)
                arr2[xxx][yyy] = int(w1.copy()[xxx][yyy], 16)
    


    

        for mm in range(0,len(arr1)):
            for nn in range(0,len(arr2)):
                state1[mm][nn] = arr1[mm][nn] ^ arr2[mm][nn] 
                
                state1[mm][nn] = BitVector(intVal= int(state1[mm][nn]), size = 8).get_bitvector_in_hex()

        #print(state1)

        #step4: inverse Mix columns
        if(i!=36):
            res =[[0 for t in range(len(Mixer))] for u in range(len(state0))]
            AES_modulus = BitVector(bitstring='100011011')
            for p in range (len(InvMixer)):
                for e in range(len(state1[0])):
                    for l in range(len(state1)):
                        v1 = InvMixer[p][l]
                        v2 = BitVector(hexstring = state1[l][e])
                        m = v1.gf_multiply_modular(v2,AES_modulus,8)
                        res[p][e] = res[p][e] ^ int(m.get_bitvector_in_hex(),16)


            for p in range(len(res)):
                for e in range(len(res[p])):
                     state1[p][e] = BitVector(intVal= res[p][e], size = 8).get_bitvector_in_hex()

        #print(state1)
        end4 = time()
        dtime = dtime + (end4-start4)
    #Deciphering is done!
    
    state1 = np.array(state1.copy()).T.tolist()
    for i in range(len(state1)):
        for j in range(len(state1[1])):
            decipher = decipher+state1[i][j]


print("The hexadecimal of the main file is: ")
print(htext)
  
print("The ciphertext is: ")
print(cipher)

print("The deciphertext is: ")
print(decipher)



# if(decipher is htext):
#     print("Successfully deciphered")
print("The main text/file was: ")
got = convertingTotext(decipher, choice)
print(got)

print("Execution time:")
print("Key scheduling: ")
print(ktime)
print("Encryption: ")
print(etime)
print("Decryption: ")
print(dtime)


# # Sbox - InvSbox generating

# In[24]:


sbox =[[0 for i in range(16)] for j in range(16)]
AES_modulus = BitVector(bitstring='100011011')
bv = BitVector(hexstring="63")
for i in range(len(sbox)):
    for j in range(len(sbox[0])):
        if (i==0 and j ==0):
            sbox[i][j] = BitVector(hexstring="63").get_bitvector_in_hex()
        else:
            hex_string = ""
            hex_string = format(i,'x')+ format(j,'x')
            b1 = BitVector(hexstring=hex_string)
            b = b1.gf_MI(AES_modulus,8)
            s = bv ^ b ^ (b<<1) ^ (b<<1) ^ (b<<1) ^ (b<<1)
            sbox[i][j] = s.get_bitvector_in_hex()
print(np.asarray(sbox))
        
        


# In[25]:


invsbox =[[0 for i in range(16)] for j in range(16)]
AES_modulus = BitVector(bitstring='100011011')
bv = BitVector(hexstring="05")
for i in range(len(sbox)):
    for j in range(len(sbox[0])):
        if (i==6 and j ==3):
            invsbox[i][j] = BitVector(hexstring="00").get_bitvector_in_hex()
        else:
            hex_string = ""
            hex_string = format(i,'x')+ format(j,'x')
            s = BitVector(hexstring=hex_string)
            b = bv ^ (s<<1) ^ (s<<2) ^ (s<<3)
            bx = b.gf_MI(AES_modulus,8)
            
            invsbox[i][j] = bx.get_bitvector_in_hex()
print(np.asarray(invsbox))


# In[ ]:




