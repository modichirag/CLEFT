import numpy as np
import cleftpool as cpool
import argparse

def parse_arguments():
   parser = argparse.ArgumentParser()
   parser.add_argument('--pfile', required=True, help='Input power spectrum in order - k, p')
   parser.add_argument('--qfile', default = None, help='File with table of Q functions in order - k, Q1, Q2, Q3, Q5, Q8, Qs2')
   parser.add_argument('--rfile', default = None, help='File with table of R functions in order - k, R1, R2')
   parser.add_argument('--npool', default = 32, type = int, help ="Number of processors")
   parser.add_argument('--nk', default = 50, type = int, help ="Number of k-values to estimate power spectrum at")
   parser.add_argument('--z', default = 0, type = float, help ="Redshift")
   parser.add_argument('--kmin', default = 1e-3, type = float, help ="minimum k")
   parser.add_argument('--kmax', default = 3, type = float, help ="maximum k")
   parser.add_argument('--M', default = 0.3, type = float, help ="Matter density")
   parser.add_argument('--outfile', default = 'pk', help ="output file")
   parser.add_argument('--saveq', default = False, type = bool, help ="Save Q kernels; saved in the directory of input spectrum")
   parser.add_argument('--saver', default = False, type = bool, help ="Save R kernels; saved in the directory of input spectrum")
   parser.add_argument('--saveqfunc', default = False, type = bool, help ="Save intermediate q-functions created; saved in the directory of input spectrum")
   parser.add_argument('--order', default=2, type=np.int, help='Order of PT')
   args = parser.parse_args()

   return args


############
if __name__=="__main__":
    
   args = parse_arguments()

   if args.pfile == None:
      print('Need an input power spectrum')

   else:

      #In no output file name is given, create from input file
      if args.outfile == 'pk':
         s = args.pfile
         fmtpos = s.rfind('.')
         outfile = s[:fmtpos] + '_cleftpk_z%03d'%(args.z*100) + s[fmtpos:]
         print('Spectrum will be saved at \n%s'%outfile)
         if args.saveq:
            qfile = s[:fmtpos] + '_cleftQ_z%03d'%(args.z*100) + s[fmtpos:]
         if args.saver:
            rfile = s[:fmtpos] + '_cleftR_z%03d'%(args.z*100) + s[fmtpos:]
         if args.saveqfunc:
            qfuncfile = s[:fmtpos] + '_cleftqfunc_z%03d'%(args.z*100) + s[fmtpos:]
         else:
            qfuncfile = None

      #Create kernels
      cl = cpool.CLEFT(pfile = args.pfile, npool = args.npool, qfile = args.qfile, rfile = args.rfile, saveqfile=qfuncfile, order=args.order);
      ##Save them
      if args.saveq:
         header = 'k[h/Mpc]   Q1   Q2    Q3    Q5    Q8     Qs2    \n'
         np.savetxt(qfile, np.array([cl.qf.kq, cl.qf.Q1, cl.qf.Q2, cl.qf.Q3, cl.qf.Q5, cl.qf.Q8, cl.qf.Qs2]).T, fmt='%0.4e', header=header)
      if args.saver:
         header = 'k[h/Mpc]   R1   R2 \n'
         np.savetxt(rfile, np.array([cl.qf.kr, cl.qf.R1, cl.qf.R2]).T, fmt='%0.4e', header=header)

      #Calculate spectrum
      pk = cpool.make_table(cl, nk = args.nk, kmin = args.kmin, kmax = args.kmax, npool = args.npool, z = args.z, M = args.M, order=args.order)

      ##Save it
      header = "k[h/Mpc]   P_Zel   P_A    P_W    P_d    P_dd     P_d^2    P_d^2d^2  P_dd^2    P_s^2    P_ds^2    P_d^2s^2   P_s^2s^2   P_D2d     P_dD2d\n"
      np.savetxt(outfile, pk, fmt='%0.4e', header=header)

      
