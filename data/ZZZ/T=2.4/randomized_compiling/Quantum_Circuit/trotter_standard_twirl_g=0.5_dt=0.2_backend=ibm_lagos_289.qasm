OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[3];
rz(-pi) q[0];
sx q[0];
rz(0.42975437) q[0];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(2.7118383) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.055145176) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(2.6864475) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-2.5864475) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.55514515) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.4365875) q[1];
sx q[1];
rz(-1.7050052) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.7050052) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.7050052) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.69374077) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.6478519) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(1.1937408) q[0];
sx q[1];
rz(1.1937408) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.2608403) q[1];
sx q[1];
rz(1.8807525) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.2608403) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.2608402) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(-pi) q[2];
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
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.92588571) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.52588575) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-0.42588571) q[0];
sx q[1];
rz(2.7157069) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.4365875) q[1];
sx q[1];
x q[1];
rz(-1.7050052) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(1.7050052) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.4365875) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.8936018) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.4479909) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.7479909) q[0];
sx q[1];
rz(1.3936018) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.0197258) q[1];
sx q[1];
rz(-pi) q[1];
rz(-1.0197257) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-2.1218669) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-2.121867) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.055145176) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.6864475) q[1];
sx q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-2.5864475) q[0];
sx q[1];
rz(-2.5864475) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.4365875) q[1];
sx q[1];
x q[1];
rz(-1.7050052) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-1.4365875) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.4365875) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.3163702) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.5163702) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(1.3252224) q[0];
sx q[1];
rz(-1.8163702) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.4437255) q[1];
sx q[1];
rz(pi) q[1];
rz(-1.4437255) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.4437255) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.6978672) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.055145176) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.6864475) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(0.55514518) q[0];
sx q[1];
rz(-2.5864475) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.4365875) q[1];
sx q[1];
rz(1.4365875) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.7050052) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.7050052) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.8507775) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-2.0908153) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(1.7908152) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.7908153) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.70985945) q[1];
sx q[1];
rz(-0.70985941) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-2.4317332) q[1];
rz(pi/2) q[2];
sx q[2];
rz(0.70985941) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.65674335) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.0567433) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.9848493) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.1567433) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.92951297) q[1];
sx q[1];
rz(pi) q[1];
rz(-0.92951295) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-2.2120797) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.92951295) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.5596469) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.7596469) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(1.0819458) q[0];
sx q[1];
rz(-2.0596469) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.018823368) q[1];
sx q[1];
rz(pi) q[1];
rz(-3.1227693) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(3.1227693) q[1];
rz(pi/2) q[2];
sx q[2];
rz(3.1227693) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
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
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.8203497) q[0];
sx q[0];
rz(pi/2) q[0];
rz(3.0628356) q[1];
sx q[1];
rz(pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-2.9628356) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.17875705) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.1479782) q[1];
sx q[1];
rz(1.1479782) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.9936145) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.1479782) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.47795235) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.8636403) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-2.1636403) q[0];
rz(-pi) q[1];
sx q[1];
rz(-2.1636403) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.0093104) q[1];
sx q[1];
rz(pi) q[1];
rz(-1.1322823) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.1322823) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-2.0093104) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[2];
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
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
