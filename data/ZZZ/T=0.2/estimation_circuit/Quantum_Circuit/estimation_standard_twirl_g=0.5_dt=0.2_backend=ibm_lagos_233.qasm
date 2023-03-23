OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4733[3];
rz(2.3877398) q[0];
sx q[0];
rz(-0.59501683) q[0];
sx q[0];
rz(-1.3564651) q[0];
rz(-1.2160185) q[1];
sx q[1];
rz(-2.1907949) q[1];
sx q[1];
rz(3.1281751) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-2.6670408) q[2];
sx q[2];
rz(-0.99073636) q[2];
sx q[2];
rz(2.3709265) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
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
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.3564651) q[0];
sx q[0];
rz(-2.5465758) q[0];
sx q[0];
rz(0.7538529) q[0];
rz(2.3709265) q[1];
sx q[1];
rz(2.1508563) q[1];
sx q[1];
rz(3.1281751) q[2];
sx q[2];
rz(-0.95079776) q[2];
sx q[2];
rz(-1.9255742) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4733[0];
measure q[1] -> c4733[1];
measure q[2] -> c4733[2];