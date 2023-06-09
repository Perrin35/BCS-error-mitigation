OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4743[3];
rz(0.29426781) q[0];
sx q[0];
rz(-2.4823967) q[0];
sx q[0];
rz(-0.35271497) q[0];
rz(1.8850243) q[1];
sx q[1];
rz(-0.94464586) q[1];
sx q[1];
rz(-0.49545728) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(0.26158456) q[2];
sx q[2];
rz(-1.6656818) q[2];
sx q[2];
rz(-2.9959909) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.35271497) q[0];
sx q[0];
rz(-2.4823967) q[0];
sx q[0];
rz(-0.29426781) q[0];
rz(-2.9959909) q[1];
sx q[1];
rz(1.4759109) q[1];
sx q[1];
rz(2.6461354) q[2];
sx q[2];
rz(-2.1969468) q[2];
sx q[2];
rz(1.2565683) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4743[0];
measure q[1] -> c4743[1];
measure q[2] -> c4743[2];
