# Almost_DIY


This repo enables to reproduce the community described in the Coordination Agent section of Almost DIY paper. By running the list of commands provided below, the proposed optimization problem can be constructed and solved. The solution will be automatically saved into a folder named as "logs". By using the generated output file, the plots in the paper can be retrieved.


nohup python -u -m energy.code.solver --num_houses 50 --mipgap 1e-2 --n_repeat 25 \
      --timelimit 900 --Q 300 >> /home/erhan/energy/out/Q300_solver_50_report_final.txt 2>&1 &
  
 nohup python -u -m energy.code.solver --num_houses 50 --mipgap 1e-2 --n_repeat 25 \
      --timelimit 900 --Q 200 >> /home/erhan/energy/out/Q200_solver_50_report_final.txt 2>&1 &
  
  nohup python -u -m energy.code.solver --num_houses 50 --mipgap 1e-2 --n_repeat 25 \
      --timelimit 900 --Q 100 >> /home/erhan/energy/out/Q100_solver_50_report_final.txt 2>&1 &
  
  
  

