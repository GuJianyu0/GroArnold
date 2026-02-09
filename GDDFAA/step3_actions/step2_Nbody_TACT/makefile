#### 这个文件先生成所有的目标文件.o, 并调用在/pot和/aa下分别包装到动态链接.so, /aa里还要链接出可执行文件.exe; 链接aa_py等??

include Makefile.inc

COORD_DIR = general/coordtransforms/
CUBA_DIR = general/cuba/
POT_DIR = pot
AA_DIR = aa
NR_DIR = general/jamestools/numrec
JT_DIR = general/jamestools/jamestools
OCTINT_DIR = general/jamestools/octint
GNUPLOT_DIR = general/gnuplot
TEST_DIR = test/
DI_DIR = DataInterface/ #gjy add

.PHONY: aa_code ct_code pot_code cuba_code octint_code jt_code nr_code gnuplot_code test

# all: cuba_code ct_code jt_code nr_code octint_code gnuplot_code pot_code aa_code
all: cuba_code ct_code jt_code nr_code octint_code gnuplot_code DataInterface_code pot_code aa_code
#gjy add: DataInterface_code #should by order

gnuplot_code:
	$(MAKE) -C $(GNUPLOT_DIR)

jt_code:
	$(MAKE) -C $(JT_DIR)

nr_code:
	$(MAKE) -C $(NR_DIR)

octint_code:
	$(MAKE) -C $(OCTINT_DIR)

cuba_code:
	$(MAKE) -C $(CUBA_DIR)

ct_code:
	$(MAKE) -C $(COORD_DIR)

DataInterface_code: #gjy add
	$(MAKE) -C $(DI_DIR)

pot_code:
	$(MAKE) -C $(POT_DIR)

aa_code:
	$(MAKE) -C $(AA_DIR)

test:
	$(MAKE) -C $(TEST_DIR)

docs:
	doxygen doc/doxygen.config

python:
	$(MAKE) -C $(AA_DIR) python
