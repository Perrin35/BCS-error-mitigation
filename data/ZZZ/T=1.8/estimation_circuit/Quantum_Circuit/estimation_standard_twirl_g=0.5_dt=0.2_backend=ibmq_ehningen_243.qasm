OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[3];
rz(0.634432262475761) q[18];
sx q[18];
rz(3.55762967203366) q[18];
sx q[18];
rz(14.0589960798769) q[18];
rz(0.301296983433917) q[21];
sx q[21];
rz(3.41971569702607) q[21];
sx q[21];
rz(13.5452395284722) q[21];
rz(5.86972614721104) q[23];
sx q[23];
rz(4.35446472130375) q[23];
sx q[23];
rz(13.5584113718754) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
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
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
rz(-pi) q[23];
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
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
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
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[21];
rz(-pi) q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
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
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
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
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[21];
rz(pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
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
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[23];
x q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
x q[23];
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
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
x q[18];
rz(-pi) q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
x q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[21];
x q[21];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
x q[21];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
rz(-pi) q[23];
barrier q[21],q[23];
cx q[21],q[23];
barrier q[21],q[23];
rz(-pi) q[23];
x q[23];
barrier q[23],q[21];
cx q[23],q[21];
barrier q[23],q[21];
x q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[18];
rz(-pi) q[21];
x q[21];
rz(-pi) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(pi) q[18];
rz(pi) q[21];
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
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[23];
x q[23];
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
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-pi) q[18];
x q[18];
rz(pi) q[21];
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
rz(-pi) q[23];
x q[23];
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
barrier q[18],q[21];
cx q[18],q[21];
barrier q[18],q[21];
rz(-pi) q[18];
x q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
rz(-4.6342181191075) q[18];
sx q[18];
rz(2.72555563514593) q[18];
sx q[18];
rz(8.79034569829362) q[18];
rz(-4.13363341110603) q[21];
sx q[21];
rz(1.92872058587584) q[21];
sx q[21];
rz(3.55505181355834) q[21];
rz(-4.1204615677028) q[23];
sx q[23];
rz(2.86346961015352) q[23];
sx q[23];
rz(9.12348097733546) q[23];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26];
measure q[18] -> c[0];
measure q[21] -> c[1];
measure q[23] -> c[2];
