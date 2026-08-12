import os
import csv

# House keeping functions
def create_folder(name):
    # print("Creating a folder: ", name)
    os.mkdir(name)

def erase_folder(Working_Folder):
    # logger.warning(" Deleting contents of {} ".format(Working_Folder))
    paths = [Working_Folder]
    inx = 0
    while (os.path.isdir(paths[inx]) and len(paths)>inx >=0):
        dirs = False
        for name in os.listdir(paths[inx]):
            path = os.path.join((paths[inx]), name)
            if (os.path.isdir(path)):
                dirs = True
                if not path in paths:
                    paths.append(path)
            else:
                # logger.debug("Removing file: {}".format(path))
                os.remove(path)
        if not dirs:
            # logger.debug("Removing dir: {}".format(paths[inx]))
            os.rmdir(paths[inx])
            paths.pop(inx)
            inx -= 1
            if inx < 0:
                break
        else:
            inx += 1

def create_tree_folder(folder, erase=None, eraseLast=False):
    tree = folder.split("/")
    path = ""
    if not erase:
        erase = [0]*len(tree)
    if len(erase) != len(tree):
        # logger.error("Erase: {} does not correspond with tree: {}".format(erase, tree))
        raise NotImplementedError("Erase: {} does not correspond with tree: {}".format(erase, tree))
    for erasei,foldy in enumerate(tree):
        path = os.path.join(path,foldy)
        if not os.path.exists(path):
            create_folder(path)
        elif erase[erasei] or (erasei + 1 == len(tree) and eraseLast):
            erase_folder(path)
            create_folder(path)

def clean_csv(input_dir, output_dir, file_list):
    create_tree_folder(output_dir)
    for file_ in file_list:
        new_file_name = file_[:-4] + "_clean.csv"

        with open(os.path.join(input_dir, file_), "r") as infile, open(os.path.join(output_dir, new_file_name), "w") as outfile:
            header = next(infile).strip() # Remove the line terminator (This is added automatically by the wirtter function)
            out_writer = csv.writer(outfile, delimiter=",", lineterminator='\n')
            in_reader = csv.reader(infile, delimiter=",")
            string = header.split(",")
            out_writer.writerow(string)
            string = ""
            for line_index, line in enumerate(in_reader):
                if len(line) < 3: # If the current line has less columns than expected (3) we shall ignore this line
                    continue
                out_writer.writerow(line)
        

        
            # Delete last line
            outfile.seek(0, os.SEEK_END)
            pos = outfile.tell()-1
            outfile.seek(pos, os.SEEK_SET)
            outfile.truncate()