OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4700[3];
rz(-3.1385848) q[0];
sx q[0];
rz(-0.042748616) q[0];
sx q[0];
rz(0.13848217) q[0];
rz(-1.1402502) q[1];
sx q[1];
rz(-1.9761338) q[1];
sx q[1];
rz(-0.15352225) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-0.94099432) q[2];
sx q[2];
rz(-1.7451844) q[2];
sx q[2];
rz(3.0674663) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
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
rz(-0.13848217) q[0];
sx q[0];
rz(-0.042748616) q[0];
sx q[0];
rz(3.1385848) q[0];
rz(0.074126356) q[1];
sx q[1];
rz(-1.7451844) q[1];
sx q[1];
rz(0.15352225) q[2];
sx q[2];
rz(-1.9761338) q[2];
sx q[2];
rz(1.1402502) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4700[0];
measure q[1] -> c4700[1];
measure q[2] -> c4700[2];
