OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(3.29812911952526) q[18];
sx q[18];
rz(3.45373316993388) q[18];
sx q[18];
rz(9.85310382549032) q[18];
rz(2.25285698776457) q[21];
sx q[21];
rz(4.36965888487077) q[21];
sx q[21];
rz(12.2915250168334) q[21];
rz(4.56122792430219) q[23];
sx q[23];
rz(4.71630099801914) q[23];
sx q[23];
rz(9.61210541331778) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-0.428325864720941) q[18];
sx q[18];
rz(2.82945213724571) q[18];
sx q[18];
rz(6.12664884124412) q[18];
rz(-0.187327452548401) q[21];
sx q[21];
rz(1.56688430916045) q[21];
sx q[21];
rz(4.86355003646719) q[21];
rz(-2.86674705606397) q[23];
sx q[23];
rz(1.91352642230882) q[23];
sx q[23];
rz(7.17192097300481) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
