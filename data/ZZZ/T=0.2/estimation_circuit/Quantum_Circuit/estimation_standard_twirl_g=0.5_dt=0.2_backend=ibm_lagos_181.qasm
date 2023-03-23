OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4681[3];
rz(3.0804816) q[0];
sx q[0];
rz(-1.1937204) q[0];
sx q[0];
rz(-0.48183577) q[0];
rz(6.0089189) q[1];
sx q[1];
rz(3.6943306) q[1];
sx q[1];
rz(9.813317) q[1];
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
rz(-pi) q[1];
x q[1];
rz(5.0027483) q[2];
sx q[2];
rz(6.2195524) q[2];
sx q[2];
rz(9.5064237) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.6597569) q[0];
sx q[0];
rz(-1.9478723) q[0];
sx q[0];
rz(0.061111048) q[0];
rz(-3.0599469) q[1];
sx q[1];
rz(-3.0779598) q[1];
sx q[1];
rz(-2.7530536) q[2];
sx q[2];
rz(-0.55273792) q[2];
sx q[2];
rz(-2.8673262) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4681[0];
measure q[1] -> c4681[1];
measure q[2] -> c4681[2];
