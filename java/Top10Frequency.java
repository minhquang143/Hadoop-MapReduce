import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Top10Frequency {
    public static class TokenizerMapper extends Mapper<Object, Text, IntWritable, IntWritable> {
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            // Đọc dòng dữ liệu từ input
            String[] parts = value.toString().split("\\s+");
            String termId = parts[0];
            int freq = Integer.parseInt(parts[2]);
            // Sử dụng termId làm key và frequency là value
            context.write(new IntWritable(Integer.parseInt(termId)), new IntWritable(freq));
        }
    }

    public static class IntSumCombiner extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            // Tính tổng frequency của termid từ tất cả các văn bản
            int totalFrequency = 0;
            for (IntWritable value : values) {
                totalFrequency += value.get();
            }

            context.write(key, new IntWritable(totalFrequency));
        }
    }

    public static class IntSumReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            // Tính tổng frequency của termid từ tất cả các văn bản
            int totalFrequency = 0;
            for (IntWritable value : values) {
                totalFrequency += value.get();
            }

            context.write(key, new IntWritable(totalFrequency));
        }
    }

    public static class DescendingKeyComparator extends WritableComparator {
        protected DescendingKeyComparator() {
            super(IntWritable.class, true);
        }
        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            IntWritable key1 = (IntWritable) w1;
            IntWritable key2 = (IntWritable) w2;
            return -1 * key1.compareTo(key2);
        }
    }

    public static class ReverseMapper extends Mapper<Object, Text, IntWritable, IntWritable> {
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            // Đọc dòng dữ liệu từ input
            String[] parts = value.toString().split("\\s+");

            IntWritable termId = new IntWritable(Integer.parseInt(parts[0]));
            IntWritable freq = new IntWritable(Integer.parseInt(parts[1]));

            context.write(freq, termId);
        }
    }

    public static class ReverseReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        static int counter = 0;
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            // reverse
            for (IntWritable val : values) {
                if (counter == 10) { return; }

                context.write(val, key);

                counter++;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf, "top 10 frequency");
        job.setJarByClass(Top10Frequency.class);

        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumCombiner.class);
        job.setReducerClass(IntSumReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1] + ".tmp"));

        job.waitForCompletion(true);

        // ------------------------
        Job job2 = Job.getInstance(conf, "filter top 10");
        job2.setJarByClass(Top10Frequency.class);

        job2.setMapperClass(ReverseMapper.class);
        job2.setReducerClass(ReverseReducer.class);

        job2.setSortComparatorClass(DescendingKeyComparator.class);

        job2.setOutputKeyClass(IntWritable.class);
        job2.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job2, new Path(args[1] + ".tmp"));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }
}