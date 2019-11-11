import csv
import pickle

def address_parse(filename)
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        address = []
        s =''
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            if line_count == 1:
                s = row[5]
                address.append(s)
                line_count += 1
                continue
            if line_count <= 10:
                s = row[2]
                s += ' '
                s += row[3]
                address.append(s)
                line_count+= 1
            else:
                break

    #filename = ('%s'%address[0])
    #outfile = open(filename, 'wb')
    #pickle.dump(address, outfile)
    #outfile.close()

    infile = open('ANAHEIM', 'rb')
    new_address = pickle.load(infile)
    infile.close()

