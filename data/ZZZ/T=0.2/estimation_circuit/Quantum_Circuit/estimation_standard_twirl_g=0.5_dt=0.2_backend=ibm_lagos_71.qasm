OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4571[3];
rz(-2.4637197) q[0];
sx q[0];
rz(-0.72655588) q[0];
sx q[0];
rz(-0.62601383) q[0];
rz(-1.9656633) q[1];
sx q[1];
rz(-2.3402218) q[1];
sx q[1];
rz(-2.3959093) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
rz(0.93456892) q[2];
sx q[2];
rz(-2.9964429) q[2];
sx q[2];
rz(-2.5341742) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
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
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.5155788) q[0];
sx q[0];
rz(-2.4150368) q[0];
sx q[0];
rz(-0.67787299) q[0];
rz(2.5341742) q[1];
sx q[1];
rz(2.9964429) q[1];
sx q[1];
rz(0.7456833) q[2];
sx q[2];
rz(-0.80137082) q[2];
sx q[2];
rz(-1.1759293) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4571[0];
measure q[1] -> c4571[1];
measure q[2] -> c4571[2];