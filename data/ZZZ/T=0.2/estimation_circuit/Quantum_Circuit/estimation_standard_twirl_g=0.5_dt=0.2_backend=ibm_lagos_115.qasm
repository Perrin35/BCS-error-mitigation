OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4615[3];
rz(0.36810045) q[0];
sx q[0];
rz(6.2129297) q[0];
sx q[0];
rz(13.708774) q[0];
rz(-0.091964713) q[1];
sx q[1];
rz(-1.8635897) q[1];
sx q[1];
rz(0.37769411) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.26270223) q[2];
sx q[2];
rz(4.3700961) q[2];
sx q[2];
rz(12.365028) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
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
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.1424036) q[0];
sx q[0];
rz(-0.070255611) q[0];
sx q[0];
rz(-0.36810045) q[0];
rz(0.20134297) q[1];
sx q[1];
rz(1.9130892) q[1];
sx q[1];
rz(0.37769411) q[2];
sx q[2];
rz(-1.2780029) q[2];
sx q[2];
rz(-3.0496279) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4615[0];
measure q[1] -> c4615[1];
measure q[2] -> c4615[2];
