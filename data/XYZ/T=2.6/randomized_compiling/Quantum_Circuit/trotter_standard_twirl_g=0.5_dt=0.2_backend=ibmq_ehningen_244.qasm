OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(-pi) q[18];
sx q[18];
rz(0.42975437) q[18];
sx q[18];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi) q[23];
sx q[23];
rz(2.7118383) q[23];
sx q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.4663074) q[18];
sx q[18];
rz(pi/2) q[18];
rz(1.2752853) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-1.1752853) q[18];
sx q[21];
rz(-1.1752853) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-0.17754317) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(2.9640495) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(0.1) q[21];
sx q[21];
rz(-pi) q[21];
rz(-3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(0.17754317) q[21];
sx q[23];
rz(-2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-0.1) q[18];
sx q[18];
rz(-0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(-pi/2) q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.5813465) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(0.76024613) q[21];
sx q[21];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(-3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(-1.0602462) q[18];
rz(-pi) q[21];
sx q[21];
rz(2.0813466) q[21];
rz(0.1) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-2.9640495) q[21];
sx q[21];
rz(pi/2) q[21];
rz(0.17754317) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
sx q[21];
rz(0.1) q[21];
sx q[21];
x q[23];
rz(-0.1) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(2.9640495) q[21];
sx q[23];
rz(2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-0.1) q[18];
sx q[18];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
sx q[21];
rz(pi/2) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(1.6752853) q[18];
sx q[18];
rz(pi/2) q[18];
rz(1.2752853) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-0.1) q[18];
sx q[18];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-1.1752853) q[18];
rz(-pi) q[21];
sx q[21];
rz(-1.1752853) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(2.9640495) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(2.9640495) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(-0.1) q[21];
sx q[21];
x q[23];
rz(0.1) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-2.9640495) q[21];
sx q[23];
rz(0.17754315) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-3.0415927) q[18];
sx q[18];
rz(-3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(1.5602462) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.76024613) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(0.1) q[18];
sx q[18];
x q[21];
rz(0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-1.0602462) q[18];
rz(-pi) q[21];
sx q[21];
rz(2.0813466) q[21];
rz(0.1) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-2.9640495) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(0.17754317) q[23];
sx q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(-0.1) q[21];
sx q[21];
x q[23];
rz(3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(2.9640495) q[21];
sx q[23];
rz(2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi/2) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(-pi) q[18];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
sx q[21];
rz(pi/2) q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.4663074) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-1.8663074) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-3.0415927) q[18];
sx q[18];
rz(-0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-1.1752853) q[18];
sx q[21];
rz(-1.1752853) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(2.9640495) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(2.9640495) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
sx q[21];
rz(0.1) q[21];
sx q[21];
x q[23];
rz(0.1) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(-2.9640495) q[21];
sx q[23];
rz(0.17754315) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(-0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(1.5602462) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(0.76024613) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(2.0813465) q[18];
sx q[21];
rz(2.0813466) q[21];
rz(0.1) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-2.9640495) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(0.17754317) q[23];
sx q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(-0.1) q[21];
sx q[21];
x q[23];
rz(3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(2.9640495) q[21];
sx q[23];
rz(2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-3.0415927) q[18];
sx q[18];
x q[21];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.4663074) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-1.8663074) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(-0.1) q[18];
sx q[18];
rz(-0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(1.9663074) q[18];
rz(-pi) q[21];
sx q[21];
rz(1.9663074) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-0.17754317) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-0.17754315) q[23];
sx q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
sx q[21];
rz(0.1) q[21];
sx q[21];
x q[23];
rz(0.1) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(0.17754317) q[21];
rz(-pi) q[23];
sx q[23];
rz(-2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(-3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
sx q[21];
rz(pi/2) q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.5813465) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-2.3813465) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
x q[21];
rz(0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-1.0602462) q[18];
sx q[21];
rz(2.0813466) q[21];
rz(0.1) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(0.17754315) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(-2.9640495) q[23];
sx q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(3.0415927) q[21];
sx q[21];
rz(-pi) q[21];
x q[23];
rz(3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(2.9640495) q[21];
rz(-pi) q[23];
sx q[23];
rz(2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi/2) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(1.6752853) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-1.8663074) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(1.9663074) q[18];
sx q[21];
rz(-1.1752853) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(2.9640495) q[21];
sx q[21];
rz(pi/2) q[21];
rz(2.9640495) q[23];
sx q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(3.0415927) q[21];
sx q[21];
rz(-3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(0.17754317) q[21];
rz(-pi) q[23];
sx q[23];
rz(-2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi/2) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.5813465) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-2.3813465) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(-0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(2.0813465) q[18];
sx q[21];
rz(2.0813466) q[21];
rz(0.1) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(0.17754315) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(0.17754317) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(-0.1) q[21];
sx q[21];
x q[23];
rz(-3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(-0.17754315) q[21];
rz(-pi) q[23];
sx q[23];
rz(2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(pi/2) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(0.1) q[18];
sx q[18];
x q[21];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
sx q[21];
rz(-pi/2) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(1.6752853) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(1.2752853) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(-3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(1.9663074) q[18];
sx q[21];
rz(-1.1752853) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(2.9640495) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(2.9640495) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(3.0415927) q[21];
sx q[21];
x q[23];
rz(0.1) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(0.17754317) q[21];
rz(-pi) q[23];
sx q[23];
rz(-2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
sx q[21];
rz(pi/2) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-1.5813465) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.76024613) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(-pi) q[18];
rz(0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(2.0813465) q[18];
sx q[21];
rz(-1.0602461) q[21];
rz(0.1) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(0.17754315) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(0.17754317) q[23];
sx q[23];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(3.0415927) q[21];
sx q[21];
rz(-pi) q[21];
x q[23];
rz(-3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(2.9640495) q[21];
sx q[23];
rz(-0.17754317) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(-pi) q[18];
x q[21];
rz(0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(1.6752853) q[18];
sx q[18];
rz(-pi/2) q[18];
rz(1.2752853) q[21];
sx q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
sx q[18];
rz(3.0415927) q[18];
sx q[18];
rz(3.0415927) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(1.9663074) q[18];
sx q[21];
rz(1.9663074) q[21];
rz(-0.3) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(2.9640495) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-0.17754315) q[23];
sx q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
sx q[21];
rz(0.1) q[21];
sx q[21];
x q[23];
rz(3.0415927) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi/2) q[21];
sx q[21];
rz(-2.9640495) q[21];
sx q[23];
rz(-2.9640495) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(pi/2) q[21];
sx q[21];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
sx q[18];
rz(0.1) q[18];
sx q[18];
rz(-pi) q[18];
rz(-0.1) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[21];
sx q[21];
rz(pi/2) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
sx q[23];
rz(pi/2) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
