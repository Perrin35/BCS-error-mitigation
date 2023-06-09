OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4772[3];
rz(1.2321826) q[0];
sx q[0];
rz(-2.4747971) q[0];
sx q[0];
rz(-2.2759721) q[0];
rz(2.5332005) q[1];
sx q[1];
rz(-1.4667888) q[1];
sx q[1];
rz(0.078740643) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
rz(0.83366771) q[2];
sx q[2];
rz(4.7250327) q[2];
sx q[2];
rz(13.516154) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.86562051) q[0];
sx q[0];
rz(-0.66679552) q[0];
sx q[0];
rz(1.90941) q[0];
rz(-2.1918088) q[1];
sx q[1];
rz(-1.58344) q[1];
sx q[1];
rz(-3.062852) q[2];
sx q[2];
rz(-1.6748039) q[2];
sx q[2];
rz(0.60839213) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4772[0];
measure q[1] -> c4772[1];
measure q[2] -> c4772[2];
